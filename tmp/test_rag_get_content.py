import asyncio
import os
from rag.graph_memory import get_graph_memory

async def test_get_content():
    gm = get_graph_memory()
    graph_id = "abitur_kb"
    
    # Let's list documents first to find a valid ID
    docs = await gm.get_list_docs(graph_id)
    if not docs:
        print("No documents found in graph.")
        return

    test_doc_id = docs[0]['id']
    print(f"Testing with doc_id: {test_doc_id}")
    
    content = await gm.get_doc_full_text(graph_id, test_doc_id)
    if content:
        print(f"Successfully retrieved content! Length: {len(content)}")
        print(f"Preview: {content[:100]}...")
    else:
        print(f"Failed to retrieve content for {test_doc_id}")

if __name__ == "__main__":
    asyncio.run(test_get_content())
