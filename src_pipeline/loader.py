
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load(self):
        loader = DirectoryLoader(
            self.data_dir,
            glob="*.md",
            loader_cls=TextLoader,
            show_progress=True
        )
        docs = loader.load()

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
                ("###", "subsection"),
            ]
        )

        chunks = []
        for doc in docs:
            parts = splitter.split_text(doc.page_content)
            for p in parts:
                p.metadata = doc.metadata
                chunks.append(p)
        return chunks
