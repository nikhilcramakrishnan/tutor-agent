from unstructured.partition.pdf import partition_pdf
from typing import List, Dict

def chunk_pdf(pdf_path: str) -> List[Dict]:
    chunks = []
    current_chunk = {"text": [], "metadata": {"page": None, "heading": None, "subheading": None}}
    
    elements = partition_pdf(pdf_path, strategy="hi_res")
    
    for elem in elements:
        page_num = getattr(elem.metadata, "page_number", 1)  # Default to 1 if not set
        
        if elem.category == "Title" and elem.text.strip().endswith(")"):
            if current_chunk["text"]:
                chunks.append({
                    "text": "\n".join(current_chunk["text"]).strip(),
                    "metadata": current_chunk["metadata"]
                })
            current_chunk = {
                "text": [elem.text],
                "metadata": {"page": page_num, "heading": elem.text.strip(), "subheading": None}
            }
            continue
        
        if elem.category == "Header" and not elem.text.strip().endswith(")"):
            if current_chunk["text"]:
                chunks.append({
                    "text": "\n".join(current_chunk["text"]).strip(),
                    "metadata": current_chunk["metadata"]
                })
            current_chunk = {
                "text": [elem.text],
                "metadata": {"page": page_num, "heading": current_chunk["metadata"]["heading"], "subheading": elem.text.strip()}
            }
            continue
        
        # Append narrative text or list items to current chunk
        if elem.category in ["NarrativeText", "ListItem"]:
            current_chunk["text"].append(elem.text)
    
    if current_chunk["text"]:
        chunks.append({
            "text": "\n".join(current_chunk["text"]).strip(),
            "metadata": current_chunk["metadata"]
        })
    
    return chunks

