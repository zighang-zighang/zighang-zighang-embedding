import glob
import numpy as np
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch

from model import KoreanEmbedding
from pypdf import PdfReader

load_dotenv()

def get_text():
    reader = PdfReader("test.pdf")

    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text()

    return all_text

if __name__ == '__main__':
    embedding_model = KoreanEmbedding()

    client = OpenSearch(
        hosts=[{'host': os.getenv('OPENSEARCH_HOST'), 'port': int(os.getenv('OPENSEARCH_PORT'))}],
        http_auth=(os.getenv('OPENSEARCH_USERNAME'), os.getenv('OPENSEARCH_PASSWORD')),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )

    search_body = {
        "size": 10,
        "query": {
            "knn": {
                "vector": {
                    "vector": embedding_model.encode(get_text()).tolist(),
                    "k": 10
                }
            }
        }
    }

    response = client.search(index='embedding_test', body=search_body)

    print(*map(lambda x: (x['_id'], x['_score']), response['hits']['hits']), sep='\n')