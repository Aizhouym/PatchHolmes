"""Online BM25+time retrieval via Elasticsearch.

Used as a fallback when precomputed bm25_time JSON files are absent.

Score formula (weights sum to 1.0):
    new_score = 0.35 * msg_bm25_norm
              + 0.15 * diff_bm25_norm
              + 0.30 * reserve_time_score
              + 0.20 * publish_time_score

BM25 is implemented via ES built-in BM25 similarity.
Two separate ES queries are issued (commit_msg / diff) so each field's
BM25 score can be individually normalised before applying weights.

Time score for a commit at rank r given CVE date at rank c:
    time_score(r, c) = 1 / (1 + 2 * |r - c|)
    (1.0 at the CVE date, decays symmetrically with commit-count distance)
"""
from __future__ import annotations

import json
import os
import glob
from pathlib import Path
from typing import Any

from patchholmes.data_models import CommitDoc, CVEQuery, RankedCandidate


# ---------------------------------------------------------------------------
# CVE date loader
# ---------------------------------------------------------------------------

def load_cve_dates(combined_csv: str | Path) -> dict[str, dict[str, str]]:
    """Return {cve_id: {reserve_time, publish_time}} from combined.csv."""
    import csv
    dates: dict[str, dict[str, str]] = {}
    with open(combined_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cve = row.get("cve", "").strip()
            if cve:
                dates[cve] = {
                    "reserve_time": row.get("reserve_time", ""),
                    "publish_time": row.get("publish_time", ""),
                }
    return dates


# ---------------------------------------------------------------------------
# ES BM25 retriever
# ---------------------------------------------------------------------------

class ESBm25Online:
    """Compute BM25+time scores online via Elasticsearch.

    Parameters
    ----------
    repo2commits_root : path to split_<repo_key> directories
    cve_dates         : dict from load_cve_dates()
    es_host           : ES host (default localhost, override via ES_HOST env)
    es_port           : ES port (default 9200, override via ES_PORT env)
    """

    MSG_WEIGHT  = 0.35
    DIFF_WEIGHT = 0.15
    RSV_WEIGHT  = 0.30
    PUB_WEIGHT  = 0.20

    def __init__(
        self,
        repo2commits_root: str | Path,
        cve_dates: dict[str, dict[str, str]],
        es_host: str | None = None,
        es_port: int | None = None,
    ) -> None:
        from elasticsearch import Elasticsearch

        self.repo2commits_root = Path(repo2commits_root)
        self.cve_dates = cve_dates

        host = es_host or os.environ.get("ES_HOST", "localhost")
        port = es_port or int(os.environ.get("ES_PORT", "9200"))
        self.es = Elasticsearch([{"host": host, "port": port, "scheme": "http"}])
        self._indexed: set[str] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: CVEQuery,
        top_k: int | None = None,
    ) -> list[RankedCandidate]:
        repo_key = query.repo_key
        cve_id   = query.cve_id
        desc     = query.description or ""

        if not desc:
            return []

        dates        = self.cve_dates.get(cve_id, {})
        reserve_time = dates.get("reserve_time", "")
        publish_time = dates.get("publish_time", "")

        try:
            self._ensure_indexed(repo_key)
        except Exception as e:
            print(f"  [ESBm25Online] indexing failed for {repo_key}: {e}", flush=True)
            return []

        idx = self._index_name(repo_key)

        try:
            msg_scores  = self._bm25_scores(idx, desc, field="commit_msg", size=10000)
            diff_scores = self._bm25_scores(idx, desc, field="diff",        size=10000)
        except Exception as e:
            print(f"  [ESBm25Online] query failed for {cve_id}: {e}", flush=True)
            return []

        all_cids = set(msg_scores) | set(diff_scores)

        # Fetch datetimes only for the candidate commits (no full-corpus scan)
        dt_map = self._fetch_datetimes(idx, list(all_cids))

        # Local rank within candidates, sorted by datetime
        sorted_cids = sorted(all_cids, key=lambda c: dt_map.get(c, ""))
        cid_rank    = {c: i for i, c in enumerate(sorted_cids)}

        reserve_rank = self._date_rank(reserve_time, sorted_cids, dt_map)
        publish_rank = self._date_rank(publish_time, sorted_cids, dt_map)

        # Normalise BM25 scores to [0, 1]
        msg_max  = max(msg_scores.values(),  default=1.0) or 1.0
        diff_max = max(diff_scores.values(), default=1.0) or 1.0

        results: list[RankedCandidate] = []
        for cid in all_cids:
            msg_norm  = msg_scores.get(cid,  0.0) / msg_max
            diff_norm = diff_scores.get(cid, 0.0) / diff_max

            r = cid_rank[cid]
            rsv_score = 1.0 / (1.0 + 2.0 * abs(r - reserve_rank))
            pub_score = 1.0 / (1.0 + 2.0 * abs(r - publish_rank))

            new_score = (
                self.MSG_WEIGHT  * msg_norm
                + self.DIFF_WEIGHT * diff_norm
                + self.RSV_WEIGHT  * rsv_score
                + self.PUB_WEIGHT  * pub_score
            )

            doc = CommitDoc(
                commit_id=cid,
                commit_msg="",
                diff="",
                owner=query.owner,
                repo=query.repo,
                datetime=dt_map.get(cid, ""),
            )
            results.append(
                RankedCandidate(
                    commit=doc,
                    score=new_score,
                    rank=0,
                    source="bm25",
                    bm25_rank=0,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        if top_k is not None:
            results = results[:top_k]
        for rank, r in enumerate(results, start=1):
            r.rank = rank
            r.bm25_rank = rank

        return results

    # ------------------------------------------------------------------
    # ES helpers
    # ------------------------------------------------------------------

    def _index_name(self, repo_key: str) -> str:
        return "patchholmes_" + repo_key.replace("@@", "__").replace("/", "_").lower()

    def _ensure_indexed(self, repo_key: str) -> None:
        if repo_key in self._indexed:
            return

        idx = self._index_name(repo_key)
        if self.es.indices.exists(index=idx):
            self._indexed.add(repo_key)
            return

        print(f"  [ESBm25Online] indexing {repo_key} into ES ...", flush=True)
        self.es.indices.create(
            index=idx,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "similarity": {"default": {"type": "BM25"}},
                },
                "mappings": {
                    "properties": {
                        "commit_id": {"type": "keyword"},
                        "commit_msg": {"type": "text"},
                        "diff":       {"type": "text"},
                        "datetime":   {"type": "keyword"},
                    }
                },
            },
        )

        from elasticsearch.helpers import bulk

        def _gen():
            split_dir = self.repo2commits_root / f"split_{repo_key}"
            for fp in glob.glob(str(split_dir / "*.json")):
                try:
                    arr = json.loads(Path(fp).read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(arr, list):
                    continue
                for item in arr:
                    cid = str(item.get("commit_id", "")).strip()
                    if not cid:
                        continue
                    yield {
                        "_index": idx,
                        "_id":    cid,
                        "_source": {
                            "commit_id":  cid,
                            "commit_msg": str(item.get("commit_msg", ""))[:8000],
                            "diff":       str(item.get("diff", ""))[:30000],
                            "datetime":   str(item.get("datetime", "")),
                        },
                    }

        bulk(self.es, _gen(), chunk_size=500, request_timeout=120)
        self.es.indices.refresh(index=idx)
        self._indexed.add(repo_key)
        print(f"  [ESBm25Online] {repo_key} indexed.", flush=True)

    def _bm25_scores(self, idx: str, query: str, field: str, size: int = 10000) -> dict[str, float]:
        """Return {commit_id: bm25_score} for top-size hits."""
        resp = self.es.search(
            index=idx,
            body={
                "query": {"match": {field: {"query": query}}},
                "size":  size,
                "_source": ["commit_id"],
            },
        )
        return {
            hit["_source"]["commit_id"]: float(hit["_score"])
            for hit in resp["hits"]["hits"]
        }

    def _fetch_datetimes(self, idx: str, cids: list[str]) -> dict[str, str]:
        """Fetch datetime for a specific set of commit IDs via terms query."""
        if not cids:
            return {}
        resp = self.es.search(
            index=idx,
            body={
                "query": {"terms": {"commit_id": cids}},
                "size":  len(cids),
                "_source": ["commit_id", "datetime"],
            },
        )
        return {
            hit["_source"]["commit_id"]: hit["_source"].get("datetime", "")
            for hit in resp["hits"]["hits"]
        }

    def _date_rank(
        self,
        date_str: str,
        sorted_cids: list[str],
        dt_map: dict[str, str],
    ) -> int:
        """Return the commit rank index closest to date_str."""
        if not date_str or not sorted_cids:
            return len(sorted_cids) // 2
        for i, cid in enumerate(sorted_cids):
            if dt_map.get(cid, "") >= date_str:
                return i
        return len(sorted_cids) - 1
