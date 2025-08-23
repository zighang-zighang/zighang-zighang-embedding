import ast
import logging
import uuid
from typing import Dict, Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from model import KoreanEmbedding

logger = logging.getLogger(__name__)


class RecruitmentsVector:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.embedding_model = KoreanEmbedding()

        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.collection_name = "recruitments"
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    @staticmethod
    def parse_company_info(company_str: str) -> Dict[str, Any]:
        try:
            if isinstance(company_str, str) and company_str.startswith('{'):
                result = ast.literal_eval(company_str)
                return result
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse company info: {company_str[:50]}... Error: {e}")

        return {}

    def create_collection(self, vector_size: int = 1024):
        logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}")

        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name in collection_names:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Successfully created collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def insert_from_csv(self, csv_file: str, batch_size: int = 16):
        logger.info(f"Starting to process jobs using batch size: {batch_size}")

        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            logger.info(f"Loaded {len(df)} jobs from {csv_file}")
        except Exception as e:
            logger.error(f"Failed to load CSV file {csv_file}: {e}")
            raise

        pushed = []
        self.create_collection()

        logger.info(f"Starting batch processing of {len(df)} jobs...")

        for i in tqdm(range(0, len(df), batch_size), desc="Processing Batches"):
            embedded = []

            for idx, row in df.iloc[i:i + batch_size].iterrows():
                try:
                    job_text = str(row.get('summaryData', '')) if pd.notna(row.get('summaryData')) else ''

                    if not job_text.strip():
                        continue

                    embedding = self.embedding_model.encode(job_text)
                    company_info = self.parse_company_info(str(row.get('company', '{}')))

                    payload = {
                        "title": str(row.get('title', '')) if pd.notna(row.get('title')) else '',
                        "company_name": company_info.get('companyName', ''),
                        "region": str(row.get('recruitmentRegion', '')) if pd.notna(
                            row.get('recruitmentRegion')) else '',
                        "address": str(row.get('recruitmentAddress', '')) if pd.notna(
                            row.get('recruitmentAddress')) else '',
                        "min_career": int(row.get('minCareer', -1)) if pd.notna(row.get('minCareer')) else -1,
                        "max_career": int(row.get('maxCareer', -1)) if pd.notna(row.get('maxCareer')) else -1,
                        "education": str(row.get('education', '')) if pd.notna(row.get('education')) else '',
                        "recruitment_deadline_type": str(row.get('recruitmentDeadlineType', '')) if pd.notna(
                            row.get('recruitmentDeadlineType')) else '',
                        "recruitment_type": str(row.get('recruitmentType', '')) if pd.notna(
                            row.get('recruitmentType')) else '',
                        "depth_one": str(row.get('depthOne', '')) if pd.notna(row.get('depthOne')) else '',
                        "depth_two": str(row.get('depthTwo', '')) if pd.notna(row.get('depthTwo')) else '',
                    }

                    embedded.append(PointStruct(
                        id=str(row.get('uuid', str(uuid.uuid4()))),
                        vector=embedding.tolist(),
                        payload=payload
                    ))
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    continue

            if embedded:
                pushed.extend(embedded)

        if pushed:
            for i in range(0, len(pushed), 100):
                chunk = pushed[i:i + 100]
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=chunk
                    )
                except Exception as e:
                    logger.error(f"Failed to insert chunk {i // 100 + 1}: {e}")
                    raise

            logger.info("Successfully inserted all vectors into Qdrant")

        logger.info("Processing completed.")

    def search(self, query_text: str, top_k: int = 5):
        try:
            query_embedding = self.embedding_model.encode(query_text)

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )

            for result in sorted(search_results, key=lambda x: x.score, reverse=True):
                print('ID:', result.id)
                print('Score:', result.score)
                print('Payload:', result.payload)
                print('---')
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise


if __name__ == "__main__":
    recruitment_vector = RecruitmentsVector()

    csv_file = "sample.csv"
    recruitment_vector.insert_from_csv(csv_file, batch_size=16)
