from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Iterable

from .models import MultiVector, SearchDocument

SCHEMA_VERSION=2

def _doc_to_dict(doc:SearchDocument)->dict:return asdict(doc)

def _doc_from_dict(data:dict)->SearchDocument:
    # Forward/backward compatibility for persisted v3 payloads.
    allowed=set(SearchDocument.__dataclass_fields__)
    clean={k:v for k,v in data.items() if k in allowed}
    vectors=clean.get("vectors")
    if isinstance(vectors,dict):clean["vectors"]=MultiVector(**vectors)
    return SearchDocument(**clean)

class SQLiteDocumentStore:
    """Thread-safe durable store with WAL, schema metadata and atomic source replacement."""
    def __init__(self,path:str|Path)->None:
        self.path=str(path);self._lock=RLock()
        self.conn=sqlite3.connect(self.path,check_same_thread=False,timeout=30.0)
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL");self.conn.execute("PRAGMA synchronous=FULL");self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY,source_id TEXT,source_version TEXT,payload TEXT NOT NULL)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS feedback (query TEXT NOT NULL,document_id TEXT NOT NULL,relevance REAL NOT NULL,kind TEXT NOT NULL,created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_document ON feedback(document_id)")
            self.conn.execute("INSERT INTO metadata(key,value) VALUES('schema_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(SCHEMA_VERSION),))
            self.conn.commit()

    @staticmethod
    def _row(d:SearchDocument)->tuple[str,str,str,str]:
        return d.id,str(d.source_id or ""),d.source_version,json.dumps(_doc_to_dict(d),ensure_ascii=False,separators=(",",":"),default=str)

    def upsert_many(self,documents:Iterable[SearchDocument],*,commit:bool=True)->None:
        rows=[self._row(d) for d in documents]
        with self._lock:
            if rows:self.conn.executemany("INSERT INTO documents(id,source_id,source_version,payload) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id,source_version=excluded.source_version,payload=excluded.payload",rows)
            if commit:self.conn.commit()

    def replace_source(self,source_id:str,documents:Iterable[SearchDocument])->None:
        rows=[self._row(d) for d in documents]
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute("DELETE FROM documents WHERE source_id=?",(str(source_id),))
                if rows:self.conn.executemany("INSERT INTO documents(id,source_id,source_version,payload) VALUES(?,?,?,?)",rows)
                self.conn.commit()
            except Exception:
                self.conn.rollback();raise

    def load_all(self)->list[SearchDocument]:
        with self._lock:rows=list(self.conn.execute("SELECT payload FROM documents"))
        return [_doc_from_dict(json.loads(row[0])) for row in rows]

    def load_source(self,source_id:str)->list[SearchDocument]:
        with self._lock:rows=list(self.conn.execute("SELECT payload FROM documents WHERE source_id=?",(str(source_id),)))
        return [_doc_from_dict(json.loads(row[0])) for row in rows]

    def ids_by_source(self,source_id:str)->list[str]:
        with self._lock:return [str(r[0]) for r in self.conn.execute("SELECT id FROM documents WHERE source_id=?",(str(source_id),))]


    def delete_source_cascade(self,source_id:str)->tuple[int,list[str]]:
        """Atomically remove a source and feedback derived from its document IDs."""
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                ids=[str(r[0]) for r in self.conn.execute("SELECT id FROM documents WHERE source_id=?",(str(source_id),))]
                if ids:
                    self.conn.executemany("DELETE FROM feedback WHERE document_id=?",[(x,) for x in ids])
                cur=self.conn.execute("DELETE FROM documents WHERE source_id=?",(str(source_id),))
                self.conn.commit();return int(cur.rowcount),ids
            except Exception:
                self.conn.rollback();raise

    def delete_source(self,source_id:str)->int:
        with self._lock:
            cur=self.conn.execute("DELETE FROM documents WHERE source_id=?",(str(source_id),));self.conn.commit();return int(cur.rowcount)

    def delete_ids(self,ids:Iterable[str])->int:
        rows=[(str(x),) for x in ids]
        with self._lock:
            count=0
            for row in rows:count+=int(self.conn.execute("DELETE FROM documents WHERE id=?",row).rowcount)
            self.conn.commit();return count

    def append_feedback(self,query:str,document_id:str,relevance:float,kind:str)->None:
        with self._lock:
            self.conn.execute("INSERT INTO feedback(query,document_id,relevance,kind) VALUES(?,?,?,?)",(query,document_id,float(relevance),kind));self.conn.commit()

    def load_feedback(self)->list[tuple[str,str,float,str]]:
        with self._lock:return [(str(q),str(d),float(r),str(k)) for q,d,r,k in self.conn.execute("SELECT query,document_id,relevance,kind FROM feedback ORDER BY rowid")]

    def delete_feedback_documents(self,ids:Iterable[str])->None:
        rows=[(str(x),) for x in ids]
        with self._lock:
            self.conn.executemany("DELETE FROM feedback WHERE document_id=?",rows);self.conn.commit()


    def get_metadata(self,key:str,default:str|None=None)->str|None:
        with self._lock:
            row=self.conn.execute("SELECT value FROM metadata WHERE key=?",(str(key),)).fetchone()
        return str(row[0]) if row else default

    def set_metadata(self,key:str,value:str)->None:
        with self._lock:
            self.conn.execute("INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(key),str(value)));self.conn.commit()

    def close(self)->None:
        with self._lock:self.conn.close()
