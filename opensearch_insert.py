import glob
import numpy as np
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()

if __name__ == '__main__':
    client = OpenSearch(
        hosts=[{'host': os.getenv('OPENSEARCH_HOST'), 'port': int(os.getenv('OPENSEARCH_PORT'))}],
        http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD')),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )

    mapping = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "innerproduct",
                        "engine": "lucene"
                    }
                }
            }
        }
    }

    if client.indices.exists(index='embedding_test'):
        client.indices.delete(index='embedding_test')

    client.indices.create(index='embedding_test', body=mapping)

    vector_files = glob.glob('output/*.npy')
    bulk_data = []
    batch_size = 500

    for path in vector_files:
        bulk_data.extend([
            {"index": {"_index": "embedding_test", "_id": os.path.splitext(os.path.basename(path))[0]}},
            {"vector": np.load(path).tolist()}
        ])

        if len(bulk_data) >= batch_size * 2:
            client.bulk(body=bulk_data)
            bulk_data = []

    if bulk_data:
        client.bulk(body=bulk_data)
