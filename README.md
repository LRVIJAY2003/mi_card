de![App Brewery Banner](https://github.com/londonappbrewery/Images/blob/master/AppBreweryBanner.png)

# Mi Card

"""
Enhanced Multimodal Retrieval-Augmented Generation (RAG) System
--------------------------------------------------------------
This comprehensive RAG system processes PDFs including text, tables, and images,
providing advanced document retrieval and summarization capabilities.

Key Features:
- Advanced PDF processing with layout-awareness
- Table, image, and chart extraction
- Multimodal content analysis
- Hybrid semantic and lexical search
- Automated document ingestion
- Streamlit web interface
- Enhanced PDF report generation
- Disk-based vector storage for scalability
- Parallel processing for performance

Usage in Vertex AI Workbench:
1. Install required packages
2. Run the setup_rag_system() function to initialize
3. Use the web interface or API functions to process queries
"""

# ===== First, install required packages =====
import sys
import subprocess
import importlib.util
import os
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Parameter .* is deprecated")

def install_packages():
    """Install or upgrade required packages."""
    packages = [
        'nltk',
        'spacy',
        'scikit-learn',
        'numpy',
        'pandas',
        'fitz',  # PyMuPDF for better PDF processing
        'tabula-py',  # For table extraction
        'opencv-python-headless',  # For image analysis
        'Pillow',  # For image processing
        'reportlab',  # For PDF generation
        'sumy',  # For fallback summarization
        'rank_bm25',  # For BM25 ranking
        'rouge',  # For evaluation
        'faiss-cpu',  # For vector storage
        'streamlit',  # For web interface
        'matplotlib',  # For visualizations
        'plotly',  # For interactive visualizations
        'datasets',  # For data handling
        'tqdm',  # For progress bars
        'pdf2image',  # For PDF to image conversion
        'pytesseract',  # For OCR
        'python-docx',  # For DOCX files (keeping for compatibility)
    ]
    
    # Optional but recommended package for better summarization
    optional_packages = [
        'sentence-transformers',
        'pdfplumber',  # Alternative PDF extraction
        'camelot-py'   # Alternative table extraction
    ]
    
    print("Checking and installing required packages...")
    
    for package in packages:
        try:
            if package == 'fitz':
                importlib.import_module('fitz')
                print(f"  ✓ PyMuPDF (fitz) already installed")
            else:
                importlib.import_module(package.replace('-', '_'))
                print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  → Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"  ✓ {package} installed successfully")
            except Exception as e:
                print(f"  ✗ Error installing {package}: {str(e)}")
    
    print("\nChecking optional packages (recommended for better performance)...")
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"  ✓ {package} already installed")
        except ImportError:
            print(f"  → Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"  ✓ {package} installed successfully")
            except Exception as e:
                print(f"  ✗ Could not install {package}: {str(e)}")

# Now import the necessary modules
import os
import re
import glob
import logging
import numpy as np
import pandas as pd
import time
import json
import cv2
import uuid
import hashlib
import pickle
import tempfile
import threading
import multiprocessing
import concurrent.futures
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional, Union, Set, Generator, Callable
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
import fitz  # PyMuPDF
import tabula
from PIL import Image
from io import BytesIO
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.platypus.flowables import Flowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
import string
import textwrap
import random
import nltk
import spacy

# Import BM25 for lexical search
from rank_bm25 import BM25Okapi

# Initialize FAISS for vector storage
try:
    import faiss
    faiss_available = True
    print("  ✓ FAISS available for efficient vector storage")
except ImportError:
    faiss_available = False
    print("  ✗ FAISS not available. Will use numpy for vector storage.")

# Download required NLTK data
print("\nDownloading required NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords, wordnet
STOPWORDS = set(stopwords.words('english'))

# Download required spaCy models
print("\nChecking spaCy models...")
try:
    # Try to load a model with word vectors if available
    try:
        nlp = spacy.load("en_core_web_md")
        print("  ✓ Using spaCy model with word vectors (en_core_web_md)")
        has_vectors = True
    except:
        print("  → Downloading spaCy model with word vectors...")
        try:
            spacy.cli.download("en_core_web_md")
            nlp = spacy.load("en_core_web_md")
            print("  ✓ Successfully downloaded and loaded en_core_web_md")
            has_vectors = True
        except:
            print("  ✗ Could not download en_core_web_md. Will use smaller model.")
            try:
                nlp = spacy.load("en_core_web_sm")
                print("  ✓ Using smaller spaCy model (en_core_web_sm)")
                has_vectors = False
            except:
                print("  → Downloading small spaCy model...")
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                print("  ✓ Successfully downloaded and loaded en_core_web_sm")
                has_vectors = False
except Exception as e:
    print(f"  ✗ Error loading spaCy models: {str(e)}")
    # Continue without stopping - we'll handle this in the code

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
    print("  ✓ Sentence Transformers available for enhanced semantic understanding")
    # Initialize the model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    sentence_transformers_available = False
    print("  ✗ Sentence Transformers not available. Will use TF-IDF for semantic similarity.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_System")

# Import sumy for traditional summarization (as fallback)
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    sumy_available = True
except ImportError:
    sumy_available = False
    print("  ✗ Sumy summarization library not available. Will use basic summarization.")

# Import rouge for evaluation
try:
    from rouge import Rouge
    rouge_available = True
    print("  ✓ Rouge metrics available for evaluation")
except ImportError:
    rouge_available = False
    print("  ✗ Rouge metrics not available. Will use basic evaluation methods.")

# Define data structures
@dataclass
class ImageData:
    """Stores information about an image extracted from a document."""
    doc_id: str
    page_num: int
    image_id: str
    image_type: str  # 'photo', 'chart', 'diagram', 'table', 'other'
    content: bytes
    width: int
    height: int
    position: Tuple[float, float, float, float]  # x0, y0, x1, y1
    caption: str = ""
    ocr_text: str = ""
    is_chart: bool = False
    is_table: bool = False
    filename: str = ""
    
    def save_to_file(self, output_dir: str) -> str:
        """Save image to file and return the path."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not self.filename:
            self.filename = f"{self.doc_id.split('.')[0]}_{self.page_num}_{self.image_id}.png"
            
        output_path = os.path.join(output_dir, self.filename)
        
        try:
            with open(output_path, 'wb') as f:
                f.write(self.content)
            return output_path
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return ""

@dataclass
class TableData:
    """Stores information about a table extracted from a document."""
    doc_id: str
    page_num: int
    table_id: str
    table_data: pd.DataFrame
    position: Tuple[float, float, float, float] = None  # x0, y0, x1, y1
    caption: str = ""
    html: str = ""
    
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        return self.table_data.to_markdown()
    
    def to_html(self) -> str:
        """Convert table to HTML format."""
        if not self.html:
            self.html = self.table_data.to_html(index=False, border=1, classes="table table-striped")
        return self.html

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    doc_id: str
    chunk_id: int
    text: str
    start_pos: int = 0
    is_title: bool = False
    is_heading: bool = False
    page_num: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Any = None
    images: List[str] = field(default_factory=list)  # List of image IDs
    tables: List[str] = field(default_factory=list)  # List of table IDs
    
    def __repr__(self):
        """String representation of the chunk."""
        return f"DocumentChunk(doc_id={self.doc_id}, chunk_id={self.chunk_id}, page={self.page_num}, len={len(self.text)}, {'title' if self.is_title else 'heading' if self.is_heading else 'text'})"
    
    def to_dict(self):
        """Convert chunk to dictionary."""
        result = asdict(self)
        result.pop('embedding', None)  # Remove embedding as it's not serializable
        return result
    
    @classmethod
    def from_dict(cls, data):
        """Create chunk from dictionary."""
        if 'embedding' in data:
            data.pop('embedding')
        return cls(**data)

@dataclass
class DocumentMetadata:
    """Stores metadata about a document."""
    doc_id: str
    title: str = ""
    author: str = ""
    creation_date: str = ""
    modification_date: str = ""
    num_pages: int = 0
    file_size: int = 0
    file_type: str = ""
    file_path: str = ""
    md5_hash: str = ""
    processing_date: str = ""
    num_images: int = 0
    num_tables: int = 0
    toc: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self):
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        """Create metadata from dictionary."""
        return cls(**data)

class PDFProcessor:
    """Advanced PDF processor with layout-awareness and multimodal extraction."""
    
    def __init__(self, output_dir: str = None, ocr_enabled: bool = True):
        """Initialize the PDF processor.
        
        Args:
            output_dir: Directory for extracted images and other assets
            ocr_enabled: Whether to perform OCR on images and scanned PDFs
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), "extracted_assets")
        self.ocr_enabled = ocr_enabled
        
        # Create output directories
        self.images_dir = os.path.join(self.output_dir, "images")
        self.tables_dir = os.path.join(self.output_dir, "tables")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        
        # Initialize image classifier (simple rule-based for now)
        self.image_classifier = ImageClassifier()
    
    def process_pdf(self, pdf_path: str) -> Tuple[str, List[DocumentChunk], DocumentMetadata, List[ImageData], List[TableData]]:
        """Process a PDF file extracting text, images, tables, and metadata."""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return "", [], None, [], []
        
        doc_id = os.path.basename(pdf_path)
        logger.info(f"Processing PDF: {doc_id}")
        
        # Extract basic metadata
        metadata = self._extract_metadata(pdf_path)
        
        # Check if PDF is mostly scanned (image-based)
        is_scanned = self._check_if_scanned(pdf_path)
        
        # Process text, images, and tables
        if is_scanned and self.ocr_enabled:
            logger.info(f"PDF appears to be scanned. Performing OCR: {doc_id}")
            text, chunks = self._process_scanned_pdf(pdf_path, doc_id)
            images = self._extract_images_from_scanned(pdf_path, doc_id)
            tables = self._extract_tables(pdf_path, doc_id)
        else:
            text, chunks, images, tables = self._process_digital_pdf(pdf_path, doc_id)
        
        # Update metadata with counts
        metadata.num_images = len(images)
        metadata.num_tables = len(tables)
        metadata.processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Link images and tables to chunks based on page number
        self._link_elements_to_chunks(chunks, images, tables)
        
        logger.info(f"Completed processing PDF: {doc_id}, {len(chunks)} chunks, {len(images)} images, {len(tables)} tables")
        
        return text, chunks, metadata, images, tables
    
    def _extract_metadata(self, pdf_path: str) -> DocumentMetadata:
        """Extract metadata from PDF."""
        doc_id = os.path.basename(pdf_path)
        
        try:
            # Calculate MD5 hash
            md5_hash = ""
            with open(pdf_path, 'rb') as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            
            # Get file info
            file_size = os.path.getsize(pdf_path)
            file_type = "PDF"
            
            # Open with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Extract basic metadata
            metadata = DocumentMetadata(
                doc_id=doc_id,
                title=doc.metadata.get("title", ""),
                author=doc.metadata.get("author", ""),
                creation_date=doc.metadata.get("creationDate", ""),
                modification_date=doc.metadata.get("modDate", ""),
                num_pages=len(doc),
                file_size=file_size,
                file_type=file_type,
                file_path=pdf_path,
                md5_hash=md5_hash
            )
            
            # Extract table of contents if available
            toc = doc.get_toc()
            if toc:
                metadata.toc = [{"level": t[0], "title": t[1], "page": t[2]} for t in toc]
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {doc_id}: {str(e)}")
            return DocumentMetadata(doc_id=doc_id, file_path=pdf_path)
    
    def _check_if_scanned(self, pdf_path: str) -> bool:
        """Check if a PDF is mostly scanned (image-based) or digital."""
        try:
            doc = fitz.open(pdf_path)
            
            # Sample a few pages
            pages_to_check = min(5, len(doc))
            page_indices = [0] + [random.randint(1, len(doc)-1) for _ in range(min(pages_to_check-1, len(doc)-1))]
            
            text_count = 0
            for page_idx in page_indices:
                page = doc[page_idx]
                text = page.get_text()
                
                # If page has reasonable amount of text, count it as a text page
                if len(text.strip()) > 100:
                    text_count += 1
            
            doc.close()
            
            # If less than 50% of sampled pages have meaningful text, consider it scanned
            return text_count < (pages_to_check / 2)
            
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {str(e)}")
            return False
    
    def _process_scanned_pdf(self, pdf_path: str, doc_id: str) -> Tuple[str, List[DocumentChunk]]:
        """Process a scanned PDF using OCR."""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            full_text = ""
            chunks = []
            chunk_id = 0
            
            for i, img in enumerate(images):
                # Perform OCR
                text = pytesseract.image_to_string(img)
                
                # Add page number to text
                page_text = f"Page {i+1}:\n{text}\n\n"
                full_text += page_text
                
                # Create chunks from OCR text
                if len(text.strip()) > 0:
                    # Simple chunking for OCR text (by paragraphs)
                    paragraphs = [p for p in text.split('\n\n') if p.strip()]
                    
                    for para in paragraphs:
                        if len(para.strip()) > 10:  # Skip very short paragraphs
                            chunks.append(DocumentChunk(
                                doc_id=doc_id,
                                chunk_id=chunk_id,
                                text=para.strip(),
                                page_num=i+1
                            ))
                            chunk_id += 1
            
            return full_text, chunks
            
        except Exception as e:
            logger.error(f"Error processing scanned PDF {doc_id}: {str(e)}")
            return "", []
    
    def _process_digital_pdf(self, pdf_path: str, doc_id: str) -> Tuple[str, List[DocumentChunk], List[ImageData], List[TableData]]:
        """Process a digital (searchable) PDF."""
        try:
            doc = fitz.open(pdf_path)
            
            full_text = ""
            chunks = []
            images = []
            tables = []
            
            # Extract tables first (so we can exclude table regions from text)
            tables = self._extract_tables(pdf_path, doc_id)
            table_regions = self._get_table_regions(tables)
            
            # Process each page
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                
                # Extract images
                page_images = self._extract_images_from_page(page, doc_id, page_idx)
                images.extend(page_images)
                
                # Get page text with layout information
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                
                # Process text blocks
                for block in blocks:
                    # Skip image blocks
                    if block["type"] == 1:  # Image block
                        continue
                        
                    # Skip blocks in table regions
                    block_bbox = (block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3])
                    if self._is_in_table_region(block_bbox, table_regions.get(page_idx+1, [])):
                        continue
                    
                    # Process text block
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"] + " "
                            page_text += "\n"
                        page_text += "\n"
                
                full_text += page_text
                
                # Create chunks from page text
                page_chunks = self._create_chunks_from_page_text(page_text, doc_id, page_idx+1)
                chunks.extend(page_chunks)
            
            doc.close()
            return full_text, chunks, images, tables
            
        except Exception as e:
            logger.error(f"Error processing digital PDF {doc_id}: {str(e)}")
            return "", [], [], []
    
    def _extract_images_from_page(self, page, doc_id: str, page_idx: int) -> List[ImageData]:
        """Extract images from a PDF page."""
        images = []
        
        try:
            # Get image list
            img_list = page.get_images(full=True)
            
            for img_idx, img_info in enumerate(img_list):
                xref = img_info[0]
                
                try:
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Try to determine image position
                    pos = (0, 0, 0, 0)  # Default position
                    for img_rect in page.get_image_rects():
                        if img_rect.get("xref") == xref:
                            pos = (img_rect["rect"][0], img_rect["rect"][1], 
                                  img_rect["rect"][2], img_rect["rect"][3])
                            break
                    
                    # Create image and analyze type
                    pil_image = Image.open(BytesIO(image_bytes))
                    width, height = pil_image.size
                    
                    # Skip very small images (likely icons or decorations)
                    if width < 50 or height < 50:
                        continue
                    
                    # Generate unique ID
                    image_id = f"img_{doc_id}_{page_idx}_{img_idx}"
                    
                    # Classify image type
                    image_type = self.image_classifier.classify_image(pil_image)
                    
                    # Create image data object
                    img_data = ImageData(
                        doc_id=doc_id,
                        page_num=page_idx+1,
                        image_id=image_id,
                        image_type=image_type,
                        content=image_bytes,
                        width=width,
                        height=height,
                        position=pos,
                        is_chart=image_type == 'chart',
                        is_table=image_type == 'table'
                    )
                    
                    # Save image to file
                    img_data.save_to_file(self.images_dir)
                    
                    # Try to extract caption (text below the image)
                    caption = self._extract_caption_near_image(page, pos)
                    img_data.caption = caption
                    
                    # If it's a chart or table image and OCR is enabled, perform OCR
                    if self.ocr_enabled and (img_data.is_chart or img_data.is_table):
                        ocr_text = pytesseract.image_to_string(pil_image)
                        img_data.ocr_text = ocr_text
                    
                    images.append(img_data)
                    
                except Exception as e:
                    logger.error(f"Error extracting image {img_idx} from page {page_idx} in {doc_id}: {str(e)}")
            
            return images
            
        except Exception as e:
            logger.error(f"Error processing images on page {page_idx} in {doc_id}: {str(e)}")
            return []
    
    def _extract_images_from_scanned(self, pdf_path: str, doc_id: str) -> List[ImageData]:
        """Extract full page images from a scanned PDF."""
        images = []
        
        try:
            # Convert PDF pages to images
            pdf_images = convert_from_path(pdf_path)
            
            for page_idx, img in enumerate(pdf_images):
                # Save image to bytes
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
                
                # Generate unique ID
                image_id = f"img_{doc_id}_{page_idx}_full"
                
                # Create image data object
                width, height = img.size
                img_data = ImageData(
                    doc_id=doc_id,
                    page_num=page_idx+1,
                    image_id=image_id,
                    image_type="scanned_page",
                    content=image_bytes,
                    width=width,
                    height=height,
                    position=(0, 0, width, height)
                )
                
                # Save image to file
                img_data.save_to_file(self.images_dir)
                images.append(img_data)
            
            return images
            
        except Exception as e:
            logger.error(f"Error extracting scanned page images from {doc_id}: {str(e)}")
            return []
    
    def _extract_caption_near_image(self, page, img_pos) -> str:
        """Extract caption near an image."""
        try:
            # Look for text below the image
            x0, y0, x1, y1 = img_pos
            
            # Define region below the image
            caption_region = fitz.Rect(x0, y1, x1, y1 + 50)  # 50 points below
            
            # Extract text from that region
            caption_text = page.get_text("text", clip=caption_region)
            
            # Clean and format caption
            caption_text = caption_text.strip()
            
            # If caption is too long, it's probably not a caption
            if len(caption_text) > 200:
                caption_text = caption_text[:197] + "..."
                
            return caption_text
            
        except Exception as e:
            logger.error(f"Error extracting caption: {str(e)}")
            return ""
    
    def _extract_tables(self, pdf_path: str, doc_id: str) -> List[TableData]:
        """Extract tables from PDF."""
        tables = []
        
        try:
            # Use tabula-py to extract tables
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            if not dfs:
                return []
            
            # Process each dataframe (table)
            for page_idx, dfs_on_page in enumerate(tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, stream=True)):
                for table_idx, df in enumerate(dfs_on_page if isinstance(dfs_on_page, list) else [dfs_on_page]):
                    if df.empty:
                        continue
                    
                    # Generate unique ID
                    table_id = f"tbl_{doc_id}_{page_idx}_{table_idx}"
                    
                    # Create table data object
                    table_data = TableData(
                        doc_id=doc_id,
                        page_num=page_idx+1,
                        table_id=table_id,
                        table_data=df,
                        position=None  # Position information not available from tabula
                    )
                    
                    # Create HTML representation
                    table_data.html = df.to_html(index=False, na_rep="", border=1)
                    
                    tables.append(table_data)
                    
                    # Also save as CSV
                    csv_path = os.path.join(self.tables_dir, f"{table_id}.csv")
                    df.to_csv(csv_path, index=False)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from {doc_id}: {str(e)}")
            return []
    
    def _get_table_regions(self, tables: List[TableData]) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """Get regions covered by tables by page."""
        table_regions = defaultdict(list)
        
        for table in tables:
            if table.position:
                table_regions[table.page_num].append(table.position)
                
        return dict(table_regions)
    
    def _is_in_table_region(self, bbox, table_regions):
        """Check if a bounding box overlaps with any table region."""
        for table_bbox in table_regions:
            # Check for overlap
            if (bbox[0] < table_bbox[2] and bbox[2] > table_bbox[0] and
                bbox[1] < table_bbox[3] and bbox[3] > table_bbox[1]):
                return True
        return False
    
    def _create_chunks_from_page_text(self, page_text: str, doc_id: str, page_num: int) -> List[DocumentChunk]:
        """Create semantic chunks from page text."""
        chunks = []
        
        if not page_text.strip():
            return chunks
        
        # Split text into lines
        lines = page_text.split('\n')
        
        # Extract title if present (first non-empty line)
        title = None
        for line in lines:
            if line.strip():
                title = line.strip()
                break
        
        # Function to identify headings
        def is_heading(line):
            # Check for heading patterns
            if re.match(r'^\s*\d+(\.\d+)*\s+\w+', line):  # Numbered heading
                return True
            if re.match(r'^\s*(Chapter|Section)\s+\d+', line, re.IGNORECASE):  # Chapter/Section
                return True
            if len(line.strip()) <= 80 and line.strip().isupper():  # ALL CAPS line
                return True
            # More patterns can be added
            return False
        
        # Identify headings
        headings = []
        for i, line in enumerate(lines):
            if is_heading(line):
                headings.append((i, line.strip()))
        
        # Create chunks based on headings
        chunk_id = 0
        last_end = 0
        
        # Add title as a special chunk if present
        if title:
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=title,
                page_num=page_num,
                is_title=True,
                metadata={"position": "title"}
            ))
            chunk_id += 1
        
        # Process each heading and its content
        for i, (heading_idx, heading_text) in enumerate(headings):
            # Get text from after the last heading to this heading
            if heading_idx > last_end:
                para_text = '\n'.join(lines[last_end:heading_idx]).strip()
                if para_text and len(para_text) > 20:
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=para_text,
                        page_num=page_num
                    ))
                    chunk_id += 1
                    
            # Add the heading itself
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=heading_text,
                page_num=page_num,
                is_heading=True
            ))
            chunk_id += 1
            
            # Determine end of section
            next_heading_idx = headings[i+1][0] if i < len(headings) - 1 else len(lines)
            
            # Add section content in chunks
            section_lines = lines[heading_idx+1:next_heading_idx]
            section_text = '\n'.join(section_lines).strip()
            
            # Split section into paragraphs
            if section_text:
                paragraphs = re.split(r'\n\s*\n', section_text)
                
                for para in paragraphs:
                    if para.strip() and len(para.strip()) > 20:
                        chunks.append(DocumentChunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=para.strip(),
                            page_num=page_num
                        ))
                        chunk_id += 1
                        
            last_end = next_heading_idx
        
        # Add any remaining text
        if last_end < len(lines):
            remaining_text = '\n'.join(lines[last_end:]).strip()
            if remaining_text and len(remaining_text) > 20:
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=remaining_text,
                    page_num=page_num
                ))
        
        return chunks
    
    def _link_elements_to_chunks(self, chunks: List[DocumentChunk], images: List[ImageData], tables: List[TableData]):
        """Link images and tables to their corresponding chunks based on page number."""
        # Create dictionaries for quick lookup
        page_images = defaultdict(list)
        page_tables = defaultdict(list)
        
        for img in images:
            page_images[img.page_num].append(img.image_id)
            
        for table in tables:
            page_tables[table.page_num].append(table.table_id)
            
        # Link to chunks
        for chunk in chunks:
            if chunk.page_num in page_images:
                chunk.images = page_images[chunk.page_num]
                
            if chunk.page_num in page_tables:
                chunk.tables = page_tables[chunk.page_num]

class ImageClassifier:
    """Simple image classifier to identify image types."""
    
    def __init__(self):
        """Initialize the image classifier."""
        pass
    
    def classify_image(self, image) -> str:
        """Classify an image as photo, chart, diagram, table, or other."""
        try:
            # Convert PIL Image to numpy array for OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale if necessary
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                
            # Rule-based classification
            # 1. Check for table characteristics (grid lines)
            horizontal_lines, vertical_lines = self._detect_lines(gray)
            if horizontal_lines > 3 and vertical_lines > 3:
                return "table"
            
            # 2. Check for chart characteristics (few colors, geometric shapes)
            if self._is_likely_chart(img_array, gray):
                return "chart"
            
            # 3. Check for diagram characteristics (line drawings, few colors)
            if self._is_likely_diagram(img_array, gray):
                return "diagram"
            
            # Default to photo if it has many colors and details
            if self._is_likely_photo(img_array):
                return "photo"
            
            # Fallback
            return "other"
            
        except Exception as e:
            logger.error(f"Error classifying image: {str(e)}")
            return "other"
    
    def _detect_lines(self, gray):
        """Detect horizontal and vertical lines."""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is None:
                return 0, 0
            
            horizontal = 0
            vertical = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Horizontal (angle close to 0 or 180)
                if angle < 10 or angle > 170:
                    horizontal += 1
                # Vertical (angle close to 90)
                elif 80 < angle < 100:
                    vertical += 1
                    
            return horizontal, vertical
            
        except Exception as e:
            logger.error(f"Error detecting lines: {str(e)}")
            return 0, 0
    
    def _is_likely_chart(self, img_array, gray):
        """Check if image is likely a chart."""
        try:
            # Charts often have:
            # 1. Limited palette of colors
            # 2. Regular shapes
            # 3. Lines or bars
            
            # Check color diversity
            if len(img_array.shape) == 3:
                # Down-sample to reduce computation
                small = cv2.resize(img_array, (64, 64))
                pixels = small.reshape(-1, 3)
                unique_colors = np.unique(pixels, axis=0)
                
                # Charts typically have a more limited color palette
                if len(unique_colors) < 30:
                    # Check for regular shapes (circles, rectangles)
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                            param1=50, param2=30, minRadius=10, maxRadius=100)
                    
                    if circles is not None and len(circles[0]) > 0:
                        return True
                    
                    # Check for horizontal lines (could be bar chart)
                    horizontal, _ = self._detect_lines(gray)
                    if horizontal > 5:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for chart: {str(e)}")
            return False
    
    def _is_likely_diagram(self, img_array, gray):
        """Check if image is likely a diagram."""
        try:
            # Diagrams often have:
            # 1. Clear edges but fewer colors than photos
            # 2. Text elements
            # 3. Basic geometric shapes
            
            # Check for edges
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # Diagrams typically have a higher ratio of edges
            if 0.05 < edge_ratio < 0.25:
                # Check color diversity (should be limited)
                if len(img_array.shape) == 3:
                    small = cv2.resize(img_array, (64, 64))
                    pixels = small.reshape(-1, 3)
                    unique_colors = np.unique(pixels, axis=0)
                    
                    # Diagrams typically have a limited color palette
                    return len(unique_colors) < 50
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for diagram: {str(e)}")
            return False
    
    def _is_likely_photo(self, img_array):
        """Check if image is likely a photo."""
        try:
            # Photos often have:
            # 1. Many different colors
            # 2. Gradients rather than sharp color transitions
            # 3. Natural textures
            
            if len(img_array.shape) != 3:
                return False
                
            # Check color diversity
            small = cv2.resize(img_array, (64, 64))
            pixels = small.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            
            # Photos typically have many colors
            return len(unique_colors) > 100
            
        except Exception as e:
            logger.error(f"Error checking for photo: {str(e)}")
            return False

class VectorDBStorage:
    """Disk-based vector storage for document embeddings."""
    
    def __init__(self, storage_dir: str):
        """Initialize the vector storage.
        
        Args:
            storage_dir: Directory to store vector indexes and metadata
        """
        self.storage_dir = storage_dir
        self.index_dir = os.path.join(storage_dir, "vector_indexes")
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        
        # Create directories
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize indexes
        self.indexes = {}
        self.metadata = {}
        self.chunk_ids = {}
        
        # Check if FAISS is available
        self.use_faiss = faiss_available
        
    def create_index(self, name: str, dimension: int):
        """Create a new vector index.
        
        Args:
            name: Index name (e.g., 'documents', 'chunks')
            dimension: Vector dimension
        """
        if name in self.indexes:
            logger.warning(f"Index {name} already exists. Will not create a new one.")
            return
            
        if self.use_faiss:
            # Create FAISS index
            index = faiss.IndexFlatL2(dimension)
            self.indexes[name] = index
            self.metadata[name] = []
            self.chunk_ids[name] = []
        else:
            # Fallback to numpy arrays
            self.indexes[name] = np.zeros((0, dimension), dtype=np.float32)
            self.metadata[name] = []
            self.chunk_ids[name] = []
            
        logger.info(f"Created vector index '{name}' with dimension {dimension}")
    
    def add_vectors(self, name: str, vectors, chunk_ids, metadata_list=None):
        """Add vectors to an index.
        
        Args:
            name: Index name
            vectors: Array of vectors to add
            chunk_ids: Corresponding chunk IDs
            metadata_list: Optional list of metadata dictionaries
        """
        if name not in self.indexes:
            # Get dimension from first vector
            if vectors.shape[0] > 0:
                dimension = vectors.shape[1]
                self.create_index(name, dimension)
            else:
                logger.error(f"Cannot add empty vectors to non-existent index {name}")
                return
        
        try:
            if self.use_faiss:
                # Convert to float32 for FAISS
                vectors = vectors.astype(np.float32)
                
                # Add to FAISS index
                self.indexes[name].add(vectors)
                
                # Store metadata and IDs
                if metadata_list:
                    self.metadata[name].extend(metadata_list)
                self.chunk_ids[name].extend(chunk_ids)
            else:
                # Add to numpy array
                self.indexes[name] = np.vstack((self.indexes[name], vectors.astype(np.float32)))
                
                # Store metadata and IDs
                if metadata_list:
                    self.metadata[name].extend(metadata_list)
                self.chunk_ids[name].extend(chunk_ids)
                
            logger.info(f"Added {len(vectors)} vectors to index '{name}'")
        except Exception as e:
            logger.error(f"Error adding vectors to index '{name}': {str(e)}")
    
    def search(self, name: str, query_vector, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Search for similar vectors in an index.
        
        Args:
            name: Index name
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of tuples (index, distance, chunk_id)
        """
        if name not in self.indexes:
            logger.error(f"Index '{name}' does not exist")
            return []
            
        try:
            if self.use_faiss:
                # Convert to float32 and reshape for FAISS
                query_vector = query_vector.astype(np.float32).reshape(1, -1)
                
                # Search FAISS index
                distances, indices = self.indexes[name].search(query_vector, top_k)
                
                # Convert to list of tuples
                results = []
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    if idx < len(self.chunk_ids[name]):
                        results.append((idx, distances[0][i], self.chunk_ids[name][idx]))
                return results
            else:
                # Search numpy array
                if self.indexes[name].shape[0] == 0:
                    return []
                    
                # Calculate distances
                distances = np.linalg.norm(self.indexes[name] - query_vector, axis=1)
                
                # Get top_k indices
                top_indices = np.argsort(distances)[:top_k]
                
                # Convert to list of tuples
                results = []
                for idx in top_indices:
                    results.append((idx, distances[idx], self.chunk_ids[name][idx]))
                return results
        except Exception as e:
            logger.error(f"Error searching index '{name}': {str(e)}")
            return []
    
    def save_to_disk(self, name: str):
        """Save an index to disk."""
        if name not in self.indexes:
            logger.error(f"Index '{name}' does not exist")
            return
            
        try:
            # Save index
            index_path = os.path.join(self.index_dir, f"{name}.idx")
            metadata_path = os.path.join(self.metadata_dir, f"{name}.pkl")
            
            if self.use_faiss:
                # Save FAISS index
                faiss.write_index(self.indexes[name], index_path)
            else:
                # Save numpy array
                np.save(index_path, self.indexes[name])
            
            # Save metadata and chunk IDs
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata[name],
                    'chunk_ids': self.chunk_ids[name]
                }, f)
                
            logger.info(f"Saved index '{name}' to disk")
        except Exception as e:
            logger.error(f"Error saving index '{name}' to disk: {str(e)}")
    
    def load_from_disk(self, name: str) -> bool:
        """Load an index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            index_path = os.path.join(self.index_dir, f"{name}.idx")
            metadata_path = os.path.join(self.metadata_dir, f"{name}.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.warning(f"Index '{name}' does not exist on disk")
                return False
            
            if self.use_faiss:
                # Load FAISS index
                self.indexes[name] = faiss.read_index(index_path)
            else:
                # Load numpy array
                self.indexes[name] = np.load(index_path)
            
            # Load metadata and chunk IDs
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata[name] = data['metadata']
                self.chunk_ids[name] = data['chunk_ids']
                
            logger.info(f"Loaded index '{name}' from disk")
            return True
        except Exception as e:
            logger.error(f"Error loading index '{name}' from disk: {str(e)}")
            return False
    
    def get_metadata(self, name: str, index: int) -> Dict[str, Any]:
        """Get metadata for a vector."""
        if name not in self.metadata or index >= len(self.metadata[name]):
            return {}
        return self.metadata[name][index]
    
    def get_indexes(self) -> List[str]:
        """Get list of available indexes."""
        return list(self.indexes.keys())

class DocumentProcessor:
    """Handles document processing, indexing, and retrieval with enhanced chunking."""
    
    def __init__(self, knowledge_base_path: str, vector_db_path: str = None, parallel_processing: bool = True):
        """Initialize the document processor."""
        logger.info(f"Initializing DocumentProcessor with path: {knowledge_base_path}")
        self.knowledge_base_path = knowledge_base_path
        self.vector_db_path = vector_db_path or os.path.join(knowledge_base_path, "vector_db")
        self.parallel_processing = parallel_processing
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path, exist_ok=True)
            
        # Create output directories
        self.assets_dir = os.path.join(knowledge_base_path, "assets")
        os.makedirs(self.assets_dir, exist_ok=True)
        
        # Initialize document storage
        self.documents = {}  # Will store document content
        self.document_chunks = {}  # Will store chunked documents
        self.document_metadata = {}  # Will store document metadata
        self.images = {}  # Will store image data
        self.tables = {}  # Will store table data
        
        # Initialize vector storage
        self.vector_db = VectorDBStorage(self.vector_db_path)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(output_dir=self.assets_dir)
        
        # Initialize vectorizer for TF-IDF fallback
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1  # Set to 1 to handle cases with few documents
        )
        
        # Initialize BM25 for lexical search
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_chunk_map = []
        
        # Track processed files to enable incremental updates
        self.processed_files = set()
        self.file_hashes = {}  # Maps file path to hash
        
        # Try to load previously processed files
        self._load_processed_files()
    
    def _load_processed_files(self):
        """Load list of previously processed files."""
        try:
            processed_files_path = os.path.join(self.knowledge_base_path, "processed_files.json")
            if os.path.exists(processed_files_path):
                with open(processed_files_path, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('files', []))
                    self.file_hashes = data.get('hashes', {})
                logger.info(f"Loaded {len(self.processed_files)} previously processed files")
        except Exception as e:
            logger.error(f"Error loading processed files: {str(e)}")
    
    def _save_processed_files(self):
        """Save list of processed files."""
        try:
            processed_files_path = os.path.join(self.knowledge_base_path, "processed_files.json")
            with open(processed_files_path, 'w') as f:
                json.dump({
                    'files': list(self.processed_files),
                    'hashes': self.file_hashes
                }, f)
        except Exception as e:
            logger.error(f"Error saving processed files: {str(e)}")
    
    def find_new_files(self) -> List[str]:
        """Find new or modified files in the knowledge base."""
        all_files = glob.glob(os.path.join(self.knowledge_base_path, "**/*.*"), recursive=True)
        
        # Filter to supported file types
        supported_extensions = ['.pdf', '.docx', '.txt']
        files = [f for f in all_files if any(f.lower().endswith(ext) for ext in supported_extensions)]
        
        # Exclude files in vector_db directory
        files = [f for f in files if not f.startswith(self.vector_db_path)]
        
        # Check for new or modified files
        new_files = []
        for file_path in files:
            # Check if file is already processed
            if file_path in self.processed_files:
                # Check if file has been modified (hash changed)
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        
                    if file_path in self.file_hashes and self.file_hashes[file_path] != file_hash:
                        new_files.append(file_path)
                        logger.info(f"File modified: {file_path}")
                except Exception as e:
                    logger.error(f"Error checking file hash: {str(e)}")
            else:
                new_files.append(file_path)
                
        logger.info(f"Found {len(new_files)} new or modified files")
        return new_files
    
    def process_all_documents(self, incremental: bool = True):
        """Process all documents including chunking and indexing."""
        # Find files to process
        if incremental:
            files_to_process = self.find_new_files()
        else:
            # Process all files in knowledge base
            files_to_process = glob.glob(os.path.join(self.knowledge_base_path, "**/*.*"), recursive=True)
            # Filter to supported file types
            supported_extensions = ['.pdf', '.docx', '.txt']
            files_to_process = [f for f in files_to_process if any(f.lower().endswith(ext) for ext in supported_extensions)]
            # Exclude files in vector_db directory
            files_to_process = [f for f in files_to_process if not f.startswith(self.vector_db_path)]
            
        if not files_to_process:
            logger.info("No new files to process")
            return
            
        logger.info(f"Processing {len(files_to_process)} documents")
        
        # Process files in parallel if enabled
        if self.parallel_processing and len(files_to_process) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
                list(tqdm(executor.map(self.process_document, files_to_process), total=len(files_to_process)))
        else:
            for file in tqdm(files_to_process):
                self.process_document(file)
                
        # Create embeddings for all chunks
        self.create_chunk_embeddings()
        
        # Create BM25 index for lexical search
        self.create_bm25_index()
        
        # Save processed files info
        self._save_processed_files()
        
        logger.info(f"Completed processing {len(files_to_process)} documents")
        logger.info(f"Total documents: {len(self.documents)}")
        logger.info(f"Total chunks: {sum(len(chunks) for chunks in self.document_chunks.values())}")
    
    def process_document(self, file_path: str) -> bool:
        """Process a single document."""
        try:
            # Get document type
            doc_id = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Calculate hash for the file
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            # Store hash
            self.file_hashes[file_path] = file_hash
            
            # Process based on file type
            if file_extension == '.pdf':
                success = self._process_pdf(file_path, doc_id)
            elif file_extension == '.docx':
                success = self._process_docx(file_path, doc_id)
            elif file_extension == '.txt':
                success = self._process_txt(file_path, doc_id)
            else:
                logger.warning(f"Unsupported file type: {file_extension} for file {doc_id}")
                return False
                
            if success:
                # Add to processed files
                self.processed_files.add(file_path)
                logger.info(f"Successfully processed document: {doc_id}")
                return True
            else:
                logger.error(f"Failed to process document: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False
    
    def _process_pdf(self, file_path: str, doc_id: str) -> bool:
        """Process a PDF document."""
        try:
            # Use PDF processor to extract text, images, tables, and metadata
            text, chunks, metadata, images, tables = self.pdf_processor.process_pdf(file_path)
            
            # Store extracted content
            self.documents[doc_id] = text
            self.document_chunks[doc_id] = chunks
            self.document_metadata[doc_id] = metadata
            
            # Store images and tables
            for img in images:
                self.images[img.image_id] = img
                
            for table in tables:
                self.tables[table.table_id] = table
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF {doc_id}: {str(e)}")
            return False
    
    def _process_docx(self, file_path: str, doc_id: str) -> bool:
        """Process a DOCX document."""
        try:
            import docx
            
            # Open the document
            doc = docx.Document(file_path)
            
            # Extract basic metadata
            metadata = DocumentMetadata(
                doc_id=doc_id,
                title=doc.core_properties.title or "",
                author=doc.core_properties.author or "",
                creation_date=str(doc.core_properties.created) if doc.core_properties.created else "",
                modification_date=str(doc.core_properties.modified) if doc.core_properties.modified else "",
                file_size=os.path.getsize(file_path),
                file_type="DOCX",
                file_path=file_path
            )
            
            # Extract text
            full_text = ""
            for para in doc.paragraphs:
                full_text += para.text + "\n"
                
            # Create chunks
            chunks = []
            chunk_id = 0
            
            # Process headings and paragraphs
            current_heading = None
            heading_start = 0
            para_idx = 0
            
            for para in doc.paragraphs:
                # Check if this is a heading
                if para.style.name.startswith('Heading'):
                    # If we had a previous heading, process its content
                    if current_heading:
                        heading_text = current_heading.text.strip()
                        if heading_text:
                            chunks.append(DocumentChunk(
                                doc_id=doc_id,
                                chunk_id=chunk_id,
                                text=heading_text,
                                is_heading=True
                            ))
                            chunk_id += 1
                            
                        # Process paragraphs under the heading
                        paras = doc.paragraphs[heading_start+1:para_idx]
                        if paras:
                            para_text = "\n".join([p.text for p in paras if p.text.strip()])
                            if para_text.strip():
                                chunks.append(DocumentChunk(
                                    doc_id=doc_id,
                                    chunk_id=chunk_id,
                                    text=para_text
                                ))
                                chunk_id += 1
                    
                    # Set current heading
                    current_heading = para
                    heading_start = para_idx
                
                para_idx += 1
            
            # Process the last heading if there was one
            if current_heading:
                heading_text = current_heading.text.strip()
                if heading_text:
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=heading_text,
                        is_heading=True
                    ))
                    chunk_id += 1
                    
                # Process paragraphs under the last heading
                paras = doc.paragraphs[heading_start+1:]
                if paras:
                    para_text = "\n".join([p.text for p in paras if p.text.strip()])
                    if para_text.strip():
                        chunks.append(DocumentChunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=para_text
                        ))
                        chunk_id += 1
            
            # Store document content
            self.documents[doc_id] = full_text
            self.document_chunks[doc_id] = chunks
            self.document_metadata[doc_id] = metadata
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing DOCX {doc_id}: {str(e)}")
            return False
    
    def _process_txt(self, file_path: str, doc_id: str) -> bool:
        """Process a TXT document."""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            # Create basic metadata
            metadata = DocumentMetadata(
                doc_id=doc_id,
                file_size=os.path.getsize(file_path),
                file_type="TXT",
                file_path=file_path
            )
            
            # Create chunks
            chunks = []
            
            # Split text into paragraphs
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            chunk_id = 0
            for para in paragraphs:
                if para.strip():
                    chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=para.strip()
                    ))
                    chunk_id += 1
                    
            # Store document content
            self.documents[doc_id] = text
            self.document_chunks[doc_id] = chunks
            self.document_metadata[doc_id] = metadata
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing TXT {doc_id}: {str(e)}")
            return False
    
    def create_chunk_embeddings(self):
        """Create embeddings for all document chunks."""
        if not self.document_chunks:
            logger.warning("No document chunks to embed")
            return
            
        logger.info("Creating embeddings for document chunks...")
        
        all_chunks = []
        chunk_texts = []
        chunk_ids = []
        
        # Collect all chunks and their texts
        for doc_id, chunks in self.document_chunks.items():
            all_chunks.extend(chunks)
            chunk_texts.extend([chunk.text for chunk in chunks])
            chunk_ids.extend([f"{doc_id}_{chunk.chunk_id}" for chunk in chunks])
        
        # Create embeddings based on available models
        if sentence_transformers_available:
            # Use sentence transformers for better semantic embeddings
            try:
                # Process in batches to avoid memory issues
                batch_size = 128
                embeddings = []
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i+batch_size]
                    batch_embeddings = sentence_model.encode(batch_texts)
                    embeddings.append(batch_embeddings)
                    
                embeddings = np.vstack(embeddings)
                
                # Store embeddings in vector DB
                self.vector_db.create_index('chunks', embeddings.shape[1])
                
                # Create metadata for each chunk
                chunk_metadata = []
                for i, chunk in enumerate(all_chunks):
                    chunk_metadata.append({
                        'doc_id': chunk.doc_id,
                        'chunk_id': chunk.chunk_id,
                        'is_title': chunk.is_title,
                        'is_heading': chunk.is_heading,
                        'page_num': chunk.page_num
                    })
                
                # Add to vector DB
                self.vector_db.add_vectors('chunks', embeddings, chunk_ids, chunk_metadata)
                
                # Save to disk
                self.vector_db.save_to_disk('chunks')
                
                # Also store in each chunk object
                for i, chunk in enumerate(all_chunks):
                    chunk.embedding = embeddings[i]
                    
            except Exception as e:
                logger.error(f"Error creating embeddings with SentenceTransformer: {str(e)}")
                # Fall back to TF-IDF
                self._create_embeddings_with_tfidf(all_chunks, chunk_texts, chunk_ids)
                
        else:
            # Fallback to TF-IDF
            self._create_embeddings_with_tfidf(all_chunks, chunk_texts, chunk_ids)
            
        logger.info(f"Created embeddings for {len(all_chunks)} chunks")
    
    def _create_embeddings_with_tfidf(self, all_chunks, chunk_texts, chunk_ids):
        """Create embeddings using TF-IDF as fallback."""
        try:
            # Fit and transform to get document embeddings
            tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            
            # Store in vector DB
            # Convert sparse matrix to dense for storage
            embeddings = tfidf_matrix.toarray()
            
            # Create vector DB index
            self.vector_db.create_index('chunks', embeddings.shape[1])
            
            # Create metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(all_chunks):
                chunk_metadata.append({
                    'doc_id': chunk.doc_id,
                    'chunk_id': chunk.chunk_id,
                    'is_title': chunk.is_title,
                    'is_heading': chunk.is_heading,
                    'page_num': chunk.page_num
                })
            
            # Add to vector DB
            self.vector_db.add_vectors('chunks', embeddings, chunk_ids, chunk_metadata)
            
            # Save to disk
            self.vector_db.save_to_disk('chunks')
            
            # Also store in each chunk object
            for i, chunk in enumerate(all_chunks):
                chunk.embedding = tfidf_matrix[i]
                
        except Exception as e:
            logger.error(f"Error creating embeddings with TF-IDF: {str(e)}")
    
    def create_bm25_index(self):
        """Create BM25 index for lexical search."""
        if not self.document_chunks:
            logger.warning("No document chunks for BM25 indexing")
            return
            
        logger.info("Creating BM25 index for document chunks...")
        
        self.bm25_corpus = []
        self.bm25_chunk_map = []
        
        # Collect all chunks and tokenize for BM25
        for doc_id, chunks in self.document_chunks.items():
            for chunk in chunks:
                # Tokenize text for BM25
                tokenized_text = nltk.word_tokenize(chunk.text.lower())
                # Remove stopwords and punctuation
                tokenized_text = [token for token in tokenized_text 
                                if token not in STOPWORDS and token not in string.punctuation]
                
                self.bm25_corpus.append(tokenized_text)
                self.bm25_chunk_map.append(chunk)
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"Created BM25 index with {len(self.bm25_corpus)} documents")
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Combine semantic and lexical search for better retrieval."""
        if not self.document_chunks:
            self.process_all_documents()
            
        if not self.document_chunks:
            logger.warning("No document chunks available for search")
            return []
            
        # Process query
        processed_query = self._preprocess_query(query)
        
        # Get semantic search results
        semantic_results = self._semantic_search(processed_query, top_k * 2)
        
        # Get BM25 lexical search results
        lexical_results = self._lexical_search(processed_query, top_k * 2)
        
        # Combine results with score normalization and reranking
        combined_results = self._combine_search_results(semantic_results, lexical_results, processed_query)
        
        # Return top_k results
        return combined_results[:top_k]
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query for search."""
        # Basic cleaning
        query = query.strip()
        
        # Remove excessive punctuation
        query = re.sub(r'([.!?])\1+', r'\1', query)
        
        return query
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Perform semantic search using vector DB."""
        try:
            # Create query embedding
            if sentence_transformers_available:
                query_embedding = sentence_model.encode(query)
            else:
                query_embedding = self.vectorizer.transform([query]).toarray()[0]
                
            # Search vector DB
            results = self.vector_db.search('chunks', query_embedding, top_k)
            
            # Convert to list of tuples (chunk, score)
            chunk_results = []
            for _, distance, chunk_id in results:
                # Parse chunk ID
                doc_id, chunk_id_str = chunk_id.split('_')
                chunk_id = int(chunk_id_str)
                
                # Find the chunk
                if doc_id in self.document_chunks:
                    for chunk in self.document_chunks[doc_id]:
                        if chunk.chunk_id == chunk_id:
                            # Convert distance to similarity score (inverse)
                            similarity = 1.0 / (1.0 + distance)
                            chunk_results.append((chunk, similarity))
                            break
            
            return chunk_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _lexical_search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """Perform lexical search using BM25."""
        if self.bm25_index is None:
            logger.warning("BM25 index not created yet")
            return []
            
        # Tokenize query
        query_tokens = nltk.word_tokenize(query.lower())
        query_tokens = [token for token in query_tokens 
                      if token not in STOPWORDS and token not in string.punctuation]
        
        # Get BM25 scores
        try:
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with chunks and scores
            results = []
            for i, score in enumerate(scores):
                results.append((self.bm25_chunk_map[i], score))
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in lexical search: {str(e)}")
            return []
    
    def _combine_search_results(self, 
                                semantic_results: List[Tuple[DocumentChunk, float]], 
                                lexical_results: List[Tuple[DocumentChunk, float]],
                                query: str) -> List[Tuple[DocumentChunk, float]]:
        """Combine and rerank semantic and lexical search results."""
        # Normalize scores to [0, 1] range
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            min_semantic = min(score for _, score in semantic_results)
            score_range = max_semantic - min_semantic
            if score_range > 0:
                semantic_results = [
                    (chunk, (score - min_semantic) / score_range) 
                    for chunk, score in semantic_results
                ]
        
        if lexical_results:
            max_lexical = max(score for _, score in lexical_results)
            min_lexical = min(score for _, score in lexical_results)
            score_range = max_lexical - min_lexical
            if score_range > 0:
                lexical_results = [
                    (chunk, (score - min_lexical) / score_range) 
                    for chunk, score in lexical_results
                ]
        
        # Combine results with weights
        combined = {}
        
        # Default weights
        semantic_weight = 0.7
        lexical_weight = 0.3
        
        # Adjust weights based on query characteristics
        if '?' in query:  # Questions often benefit from semantic search
            semantic_weight = 0.8
            lexical_weight = 0.2
        elif any(term in query.lower() for term in ['how', 'why', 'explain']):
            semantic_weight = 0.8
            lexical_weight = 0.2
        elif len(query.split()) <= 3:  # Short queries often benefit from lexical search
            semantic_weight = 0.5
            lexical_weight = 0.5
            
        # Add semantic results
        for chunk, score in semantic_results:
            chunk_id = (chunk.doc_id, chunk.chunk_id)
            combined[chunk_id] = {"chunk": chunk, "score": score * semantic_weight}
            
        # Add lexical results
        for chunk, score in lexical_results:
            chunk_id = (chunk.doc_id, chunk.chunk_id)
            if chunk_id in combined:
                combined[chunk_id]["score"] += score * lexical_weight
            else:
                combined[chunk_id] = {"chunk": chunk, "score": score * lexical_weight}
        
        # Apply additional reranking factors
        for chunk_id, data in combined.items():
            chunk = data["chunk"]
            
            # Boost titles and headings
            if chunk.is_title:
                data["score"] += 0.2
            elif chunk.is_heading:
                data["score"] += 0.1
                
            # Boost chunks with tables/images if query mentions them
            if (chunk.tables or chunk.images) and any(term in query.lower() for term in ['table', 'chart', 'figure', 'image', 'picture', 'graph']):
                data["score"] += 0.15
                
            # Cap at 1.0
            data["score"] = min(1.0, data["score"])
        
        # Convert to list and sort
        results = [(data["chunk"], data["score"]) for data in combined.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def keyword_search(self, keyword: str) -> List[Tuple[str, DocumentChunk]]:
        """Search for keyword matches in document chunks."""
        if not self.document_chunks:
            self.process_all_documents()
            
        if not self.document_chunks:
            logger.warning("No document chunks for keyword search")
            return []
            
        results = []
        keyword_lower = keyword.lower()
        
        for doc_id, chunks in self.document_chunks.items():
            for chunk in chunks:
                if keyword_lower in chunk.text.lower():
                    results.append((doc_id, chunk))
        
        return results
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        if not text or len(text.strip()) == 0:
            return []
            
        try:
            doc = nlp(text[:50000])  # Limit text length to avoid memory issues
            
            # Extract noun phrases and entities
            key_terms = []
            
            # Get named entities
            for ent in doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT', 'GPE', 'PERSON', 'WORK_OF_ART', 'EVENT'):
                    key_terms.append(ent.text)
            
            # Get noun phrases
            for chunk in doc.noun_chunks:
                # Only include multi-word phrases or important single nouns
                if len(chunk.text.split()) > 1 or (chunk.root.pos_ == 'NOUN' and chunk.root.tag_ not in ('NN', 'NNS')):
                    key_terms.append(chunk.text)
            
            # Remove duplicates and sort by length (longer terms first)
            key_terms = list(set(key_terms))
            key_terms.sort(key=lambda x: len(x), reverse=True)
            
            return key_terms[:7]  # Return top 7 terms
            
        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}")
            return []
    
    def get_image_by_id(self, image_id: str) -> Optional[ImageData]:
        """Get image by ID."""
        return self.images.get(image_id)
    
    def get_table_by_id(self, table_id: str) -> Optional[TableData]:
        """Get table by ID."""
        return self.tables.get(table_id)
    
    def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID."""
        return self.document_metadata.get(doc_id)

class QueryProcessor:
    """Enhanced query processing with expansion and reformulation."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.nlp = nlp
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query to extract key concepts and expand with related terms.
        
        Returns a dictionary with:
        - original_query: The original query string
        - expanded_query: Query expanded with synonyms and related terms
        - keywords: Important keywords extracted from the query
        - entities: Named entities in the query
        - query_type: Type of query (question, command, etc.)
        - question_type: For questions, what kind of question it is
        """
        query = query.strip()
        
        # Basic processing
        result = {
            "original_query": query,
            "expanded_query": query,
            "keywords": [],
            "entities": [],
            "query_type": "unknown",
            "question_type": None
        }
        
        # Parse the query
        doc = self.nlp(query)
        
        # Extract query type
        if query.endswith('?'):
            result["query_type"] = "question"
        elif query.endswith('!'):
            result["query_type"] = "command"
        elif query.lower().startswith(('find', 'search', 'get', 'retrieve')):
            result["query_type"] = "search"
        elif query.lower().startswith(('tell', 'explain', 'describe', 'elaborate')):
            result["query_type"] = "explanation"
        else:
            result["query_type"] = "statement"
        
        # For questions, determine question type
        if result["query_type"] == "question":
            question_words = {
                'what': 'definition', 
                'how': 'process', 
                'why': 'reason', 
                'when': 'time', 
                'where': 'location', 
                'who': 'person',
                'which': 'selection'
            }
            
            first_word = doc[0].text.lower() if len(doc) > 0 else ""
            if first_word in question_words:
                result["question_type"] = question_words[first_word]
            else:
                result["question_type"] = "general"
        
        # Extract keywords (important nouns, verbs, and adjectives)
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN') and not token.is_stop:
                result["keywords"].append(token.text.lower())
            elif token.pos_ in ('VERB', 'ADJ') and token.is_alpha and len(token.text) > 2 and not token.is_stop:
                result["keywords"].append(token.text.lower())
        
        # Extract named entities
        for ent in doc.ents:
            result["entities"].append((ent.text, ent.label_))
            
            # Add entity text to keywords if not already there
            if ent.text.lower() not in result["keywords"]:
                result["keywords"].append(ent.text.lower())
        
        # Extract visual elements references
        if any(term in query.lower() for term in ['image', 'picture', 'photo', 'figure', 'chart', 'graph', 'diagram']):
            result["has_visual_reference"] = True
        else:
            result["has_visual_reference"] = False
            
        # Extract table references
        if any(term in query.lower() for term in ['table', 'spreadsheet', 'grid', 'data']):
            result["has_table_reference"] = True
        else:
            result["has_table_reference"] = False
        
        # Expand query with synonyms and related terms
        expanded_query = self._expand_query(query, result["keywords"])
        result["expanded_query"] = expanded_query
        
        return result
    
    def _expand_query(self, original_query: str, keywords: List[str]) -> str:
        """Expand query with synonyms and related terms."""
        if not keywords:
            return original_query
            
        expansion_terms = set()
        
        # Add synonyms for important keywords
        for keyword in keywords[:3]:  # Limit to top 3 keywords to avoid over-expansion
            synonyms = self._get_synonyms(keyword)
            # Add up to 2 synonyms per keyword
            expansion_terms.update(synonyms[:2])
        
        # Remove original keywords from expansion terms
        expansion_terms = expansion_terms - set(keywords)
        
        # Limit to top 5 expansion terms
        expansion_terms = list(expansion_terms)[:5]
        
        if not expansion_terms:
            return original_query
            
        # Create expanded query
        expanded_query = original_query + " " + " ".join(expansion_terms)
        
        return expanded_query
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        # Look for synonyms in WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().lower().replace('_', ' ')
                # Only add if different from original and a single word
                if synonym != word and len(synonym.split()) == 1:
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def rewrite_query(self, query: str, context: List[str] = None) -> str:
        """
        Rewrite an ambiguous query to be more specific using context if available.
        
        Args:
            query: The original query
            context: Optional list of previous queries or responses for context
            
        Returns:
            Rewritten query if clarification needed, otherwise original query
        """
        # Only rewrite certain types of ambiguous queries
        if len(query.split()) > 5:
            # Longer queries are usually specific enough
            return query
            
        # Check for ambiguous pronouns without context
        doc = self.nlp(query)
        has_pronoun = any(token.pos_ == 'PRON' for token in doc)
        
        if has_pronoun and not context:
            # Ambiguous pronouns without context - leave as is and let the
            # system handle it as best it can
            return query
            
        # If we have context and pronouns, try to resolve them
        if has_pronoun and context:
            # Basic pronoun resolution using last context
            last_context = context[-1]
            
            # Extract potential entities from the last context
            last_doc = self.nlp(last_context)
            entities = [ent.text for ent in last_doc.ents]
            
            if entities:
                # Replace common pronouns with the most recent entity
                # This is a simplistic approach - a real system would use proper coreference resolution
                replacements = {
                    'it': entities[-1],
                    'this': entities[-1],
                    'that': entities[-1],
                    'they': entities[-1],
                    'them': entities[-1],
                    'these': entities[-1],
                    'those': entities[-1]
                }
                
                words = query.split()
                for i, word in enumerate(words):
                    lower_word = word.lower()
                    if lower_word in replacements:
                        words[i] = replacements[lower_word]
                        
                return ' '.join(words)
                
        return query

class SemanticProcessor:
    """Handles semantic processing, including similarity calculations and embeddings."""
    
    def __init__(self):
        """Initialize the semantic processor with the appropriate models."""
        if sentence_transformers_available:
            # Use sentence transformers for better semantic understanding
            self.model = sentence_model
            logger.info("Using SentenceTransformer for semantic processing")
        else:
            # Fall back to TF-IDF
            self.vectorizer = TfidfVectorizer(stop_words='english')
            logger.info("Using TF-IDF for semantic processing")
    
    def get_text_embedding(self, text: str):
        """Get embedding for a text."""
        if not text or len(text.strip()) == 0:
            return None
            
        try:
            if sentence_transformers_available:
                return self.model.encode(text)
            else:
                # Use TF-IDF as fallback
                return self.vectorizer.fit_transform([text])[0]
        except Exception as e:
            logger.error(f"Error getting text embedding: {str(e)}")
            return None
    
    def get_embeddings(self, texts: List[str]):
        """Get embeddings for multiple texts."""
        if not texts:
            return []
            
        try:
            if sentence_transformers_available:
                return self.model.encode(texts)
            else:
                # Use TF-IDF as fallback
                return self.vectorizer.fit_transform(texts)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        try:
            if sentence_transformers_available:
                embedding1 = self.model.encode(text1)
                embedding2 = self.model.encode(text2)
                # Compute cosine similarity
                return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            else:
                # Use TF-IDF and cosine similarity as fallback
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate similarity matrix for a list of texts."""
        if not texts:
            return np.array([])
            
        try:
            if sentence_transformers_available:
                embeddings = self.model.encode(texts)
                # Compute pairwise cosine similarities
                similarity_matrix = np.zeros((len(texts), len(texts)))
                for i in range(len(texts)):
                    for j in range(i, len(texts)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                return similarity_matrix
            else:
                # Use TF-IDF and cosine similarity as fallback
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                return cosine_similarity(tfidf_matrix)
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {str(e)}")
            return np.zeros((len(texts), len(texts)))

class ImprovedSummaryGenerator:
    """Generates concise, coherent, non-repetitive summaries from retrieved documents."""
    
    def __init__(self):
        """Initialize the improved summary generator."""
        self.language = "english"
        self.stop_words = STOPWORDS
        self.semantic_

class ImprovedSummaryGenerator:
    """Generates concise, coherent, non-repetitive summaries from retrieved documents."""
    
    def __init__(self):
        """Initialize the improved summary generator."""
        self.language = "english"
        self.stop_words = STOPWORDS
        self.semantic_processor = SemanticProcessor()
        
        # If sumy is available, initialize its components for fallback
        if sumy_available:
            self.stemmer = Stemmer(self.language)
            self.sumy_stop_words = get_stop_words(self.language)
            
            # Initialize summarizers
            self.lexrank = LexRankSummarizer(self.stemmer)
            self.lexrank.stop_words = self.sumy_stop_words
            
            self.lsa = LsaSummarizer(self.stemmer)
            self.lsa.stop_words = self.sumy_stop_words
    
    def generate_summary(self, 
                         chunks: List[DocumentChunk], 
                         query_info: Dict[str, Any], 
                         images: List[ImageData] = None,
                         tables: List[TableData] = None,
                         max_length: int = 500) -> Dict[str, Any]:
        """
        Generate a concise, non-repetitive summary from document chunks relevant to the query.
        
        Args:
            chunks: List of document chunks to summarize
            query_info: Query information from QueryProcessor
            images: List of images related to retrieved chunks
            tables: List of tables related to retrieved chunks
            max_length: Target maximum length of summary in words
            
        Returns:
            Dictionary with summary text and references to images/tables
        """
        if not chunks:
            return {
                "summary": "No relevant information found.",
                "referenced_images": [],
                "referenced_tables": []
            }
            
        try:
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Process and clean the texts
            processed_texts = self._preprocess_texts(texts)
            if not processed_texts:
                return {
                    "summary": "No relevant information found after processing.",
                    "referenced_images": [],
                    "referenced_tables": []
                }
                
            # Extract query intent and key concepts from query_info
            query_concepts = self._extract_key_concepts_from_query(query_info)
            
            # Extract information units (sentences or paragraphs)
            information_units = self._extract_information_units(processed_texts, chunks)
            
            # Score information units by relevance to query
            scored_units = self._score_by_query_relevance(information_units, query_concepts, query_info["original_query"])
            
            # Remove redundant information
            unique_units = self._remove_redundancies(scored_units)
            
            # Select top units within target length
            selected_units = self._select_within_length(unique_units, max_length)
            
            # Organize information logically
            organized_units = self._organize_information(selected_units, query_concepts)
            
            # Find relevant images and tables
            referenced_images = []
            referenced_tables = []
            
            if images or tables:
                referenced_images, referenced_tables = self._find_relevant_media(
                    organized_units, query_info, images, tables
                )
            
            # Generate the final summary
            summary = self._generate_coherent_text(organized_units, query_info["original_query"], query_concepts)
            
            return {
                "summary": summary,
                "referenced_images": referenced_images,
                "referenced_tables": referenced_tables
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fall back to traditional summarization if available
            if sumy_available:
                fallback_summary = self._generate_fallback_summary(texts, query_info["original_query"])
                return {
                    "summary": fallback_summary,
                    "referenced_images": [],
                    "referenced_tables": []
                }
            else:
                return {
                    "summary": f"An error occurred while generating the summary. Please try again with a different query.",
                    "referenced_images": [],
                    "referenced_tables": []
                }
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Clean and normalize the input texts."""
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                continue
                
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove excessive punctuation
            text = re.sub(r'([.!?])\1+', r'\1', text)
            
            # Remove content in brackets if it looks like citations or references
            text = re.sub(r'\[\d+\]', '', text)
            
            processed_texts.append(text.strip())
            
        return processed_texts
    
    def _extract_key_concepts_from_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key concepts from the query info."""
        concepts = {
            'keywords': query_info.get('keywords', []),
            'entities': query_info.get('entities', []),
            'query_type': query_info.get('query_type', 'unknown'),
            'question_type': query_info.get('question_type'),
            'original_query': query_info.get('original_query', ''),
            'has_visual_reference': query_info.get('has_visual_reference', False),
            'has_table_reference': query_info.get('has_table_reference', False)
        }
        
        return concepts
    
    def _extract_information_units(self, texts: List[str], chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Extract meaningful information units from texts."""
        # Extract units along with their source chunk information
        all_units = []
        
        for i, text in enumerate(texts):
            # Get the corresponding chunk
            chunk = chunks[i] if i < len(chunks) else None
            
            # If chunk is a title or heading, keep it as a unit
            if chunk and (chunk.is_title or chunk.is_heading):
                all_units.append({
                    'text': text,
                    'chunk': chunk,
                    'is_title': chunk.is_title,
                    'is_heading': chunk.is_heading
                })
                continue
                
            # For longer texts, split by sentences
            if len(text) > 100:
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    # Skip very short sentences
                    if len(sent.split()) < 5:
                        continue
                        
                    all_units.append({
                        'text': sent,
                        'chunk': chunk,
                        'is_title': False,
                        'is_heading': False
                    })
            else:
                # For shorter texts, use as is
                all_units.append({
                    'text': text,
                    'chunk': chunk,
                    'is_title': False,
                    'is_heading': False
                })
                
        # Filter out units with little information
        filtered_units = []
        for unit in all_units:
            # Always keep titles and headings
            if unit['is_title'] or unit['is_heading']:
                filtered_units.append(unit)
                continue
                
            # Skip units that are mostly stopwords
            tokens = unit['text'].lower().split()
            if not tokens:
                continue
                
            non_stop_ratio = sum(1 for t in tokens if t not in self.stop_words) / len(tokens)
            if non_stop_ratio < 0.4:
                continue
                
            filtered_units.append(unit)
            
        return filtered_units
    
    def _score_by_query_relevance(self, 
                                  units: List[Dict[str, Any]], 
                                  query_concepts: Dict[str, Any],
                                  original_query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Score information units by relevance to the query concepts."""
        if not units:
            return []
            
        scored_units = []
        
        # Convert query keywords to a space-separated string for comparison
        query_text = ' '.join(query_concepts['keywords']) if query_concepts['keywords'] else original_query
        
        # Use the semantic processor to calculate relevance
        for unit in units:
            # Base score on semantic similarity
            similarity = self.semantic_processor.calculate_similarity(query_text, unit['text'])
            
            # Apply additional scoring based on query concepts
            bonus = 0
            
            # Boost titles and headings
            if unit['is_title']:
                bonus += 0.3
            elif unit['is_heading']:
                bonus += 0.2
            
            # Check for exact entity matches
            for entity, _ in query_concepts['entities']:
                if entity.lower() in unit['text'].lower():
                    bonus += 0.2
            
            # Check if unit contains information related to question type
            if query_concepts['question_type'] == 'definition' and any(phrase in unit['text'].lower() 
                                                                      for phrase in ['is a', 'refers to', 'defined as']):
                bonus += 0.15
            elif query_concepts['question_type'] == 'process' and any(phrase in unit['text'].lower() 
                                                                     for phrase in ['steps', 'process', 'how to']):
                bonus += 0.15
                
            # Boost units with images/tables if query has visual/table references
            if query_concepts.get('has_visual_reference', False) and unit['chunk'] and unit['chunk'].images:
                bonus += 0.25
                
            if query_concepts.get('has_table_reference', False) and unit['chunk'] and unit['chunk'].tables:
                bonus += 0.25
                
            # Apply bonuses (capped at 1.0)
            final_score = min(1.0, similarity + bonus)
            scored_units.append((unit, final_score))
                
        # Sort by score in descending order
        scored_units.sort(key=lambda x: x[1], reverse=True)
        return scored_units
    
    def _remove_redundancies(self, scored_units: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Remove redundant information to avoid repetition."""
        if not scored_units:
            return []
            
        # Extract units and scores
        units = [item[0]['text'] for item in scored_units]
        original_items = [item for item in scored_units]
        
        # Calculate similarity matrix between all units
        similarity_matrix = self.semantic_processor.calculate_similarity_matrix(units)
        
        # Initialize list to keep track of which units to keep
        keep_mask = np.ones(len(units), dtype=bool)
        
        # Iterate through units by score (already sorted)
        for i in range(len(units)):
            if not keep_mask[i]:
                continue  # Skip if already marked for removal
                
            # Check this unit against all subsequent (lower-scored) units
            for j in range(i + 1, len(units)):
                if not keep_mask[j]:
                    continue  # Skip if already marked for removal
                    
                # If units are too similar, remove the lower-scored one
                if similarity_matrix[i, j] > 0.7:  # Threshold for similarity
                    keep_mask[j] = False
                    
        # Create filtered list of non-redundant units with their scores
        unique_units = [original_items[i] for i in range(len(units)) if keep_mask[i]]
        return unique_units
    
    def _select_within_length(self, scored_units: List[Tuple[Dict[str, Any], float]], max_length: int) -> List[Tuple[Dict[str, Any], float]]:
        """Select top units that fit within the target length."""
        if not scored_units:
            return []
            
        current_length = 0
        selected_units = []
        
        for unit, score in scored_units:
            unit_length = len(unit['text'].split())
            if current_length + unit_length <= max_length:
                selected_units.append((unit, score))
                current_length += unit_length
            
            if current_length >= max_length:
                break
                
        return selected_units
    
    def _organize_information(self, scored_units: List[Tuple[Dict[str, Any], float]], query_concepts: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Organize selected information into a logical structure."""
        if not scored_units:
            return []
            
        # Determine organization strategy based on query
        if query_concepts['question_type'] == 'definition':
            # For definition queries, put definitive statements first
            definitions = []
            elaborations = []
            examples = []
            
            for unit, score in scored_units:
                text = unit['text'].lower()
                if unit['is_title'] or unit['is_heading']:
                    # Titles and headings go first
                    definitions.append((unit, score))
                elif any(phrase in text for phrase in ['is a', 'refers to', 'defined as', 'meaning of']):
                    definitions.append((unit, score))
                elif 'example' in text or 'instance' in text or 'such as' in text:
                    examples.append((unit, score))
                else:
                    elaborations.append((unit, score))
                    
            # Combine in logical order: definition, elaboration, examples
            return definitions + elaborations + examples
            
        elif query_concepts['question_type'] == 'process':
            # Try to organize steps in a process
            # Look for indicators of sequential steps
            units_with_markers = []
            
            for unit, score in scored_units:
                text = unit['text'].lower()
                # Look for step indicators
                has_step = bool(re.search(r'(?:^|\W)(?:step\s*\d|first|second|third|next|then|finally)(?:\W|$)', text))
                priority = 2 if unit['is_title'] else 1 if unit['is_heading'] else 0
                units_with_markers.append((unit, score, has_step, priority))
            
            # Sort: first by priority, then by step marker, then by score
            ordered_units = sorted(units_with_markers, key=lambda x: (-x[3], -x[2], -x[1]))
            return [(unit, score) for unit, score, _, _ in ordered_units]
            
        elif query_concepts['query_type'] == 'explanation':
            # For explanations, start with overview then details
            overview = []
            details = []
            
            for unit, score in scored_units:
                if unit['is_title'] or unit['is_heading']:
                    overview.append((unit, score))
                elif len(unit['text'].split()) < 20:  # Shorter sentences often give overviews
                    overview.append((unit, score))
                else:
                    details.append((unit, score))
                    
            return overview + details
            
        else:
            # For other types, start with titles/headings, then highest relevance units
            titles = []
            others = []
            
            for unit, score in scored_units:
                if unit['is_title'] or unit['is_heading']:
                    titles.append((unit, score))
                else:
                    others.append((unit, score))
                    
            # Sort non-titles by score
            others.sort(key=lambda x: x[1], reverse=True)
            
            return titles + others
    
    def _find_relevant_media(self, 
                            organized_units: List[Tuple[Dict[str, Any], float]], 
                            query_info: Dict[str, Any],
                            images: List[ImageData],
                            tables: List[TableData]) -> Tuple[List[str], List[str]]:
        """Find relevant images and tables based on query and selected units."""
        referenced_images = []
        referenced_tables = []
        
        # Check if we have images and tables
        if not images and not tables:
            return [], []
        
        # Extract all chunks used in the summary
        used_chunks = [unit['chunk'] for unit, _ in organized_units if unit['chunk']]
        
        # First pass: collect media referenced in chunks used in summary
        for chunk in used_chunks:
            if chunk.images:
                for img_id in chunk.images:
                    if img_id not in referenced_images:
                        referenced_images.append(img_id)
                        
            if chunk.tables:
                for table_id in chunk.tables:
                    if table_id not in referenced_tables:
                        referenced_tables.append(table_id)
        
        # Second pass: check if the query specifically asks for visual content
        if query_info.get('has_visual_reference', False) and not referenced_images and images:
            # Add most relevant images based on image type
            relevant_image_types = []
            
            # Determine relevant image types from query
            query_text = query_info['original_query'].lower()
            if any(term in query_text for term in ['chart', 'graph', 'plot']):
                relevant_image_types.append('chart')
            if any(term in query_text for term in ['diagram', 'flow', 'architecture']):
                relevant_image_types.append('diagram')
            if any(term in query_text for term in ['photo', 'picture', 'image']):
                relevant_image_types.append('photo')
                
            # If no specific type mentioned, consider all types relevant
            if not relevant_image_types:
                relevant_image_types = ['chart', 'diagram', 'photo', 'table', 'other']
                
            # Find images of relevant types
            for img in images:
                if img.image_type in relevant_image_types and img.image_id not in referenced_images:
                    referenced_images.append(img.image_id)
                    # Limit to top 3 most relevant images
                    if len(referenced_images) >= 3:
                        break
        
        # Third pass: check if the query specifically asks for tables
        if query_info.get('has_table_reference', False) and not referenced_tables and tables:
            # Add up to 2 tables
            for table in tables:
                if table.table_id not in referenced_tables:
                    referenced_tables.append(table.table_id)






class ImprovedSummaryGenerator:
    """Generates concise, coherent, non-repetitive summaries from retrieved documents."""
    
    def __init__(self):
        """Initialize the improved summary generator."""
        self.language = "english"
        self.stop_words = STOPWORDS
        self.semantic_processor = SemanticProcessor()
        
        # If sumy is available, initialize its components for fallback
        if sumy_available:
            self.stemmer = Stemmer(self.language)
            self.sumy_stop_words = get_stop_words(self.language)
            
            # Initialize summarizers
            self.lexrank = LexRankSummarizer(self.stemmer)
            self.lexrank.stop_words = self.sumy_stop_words
            
            self.lsa = LsaSummarizer(self.stemmer)
            self.lsa.stop_words = self.sumy_stop_words
    
    def generate_summary(self, 
                         chunks: List[DocumentChunk], 
                         query_info: Dict[str, Any], 
                         images: List[ImageData] = None,
                         tables: List[TableData] = None,
                         max_length: int = 500) -> Dict[str, Any]:
        """
        Generate a concise, non-repetitive summary from document chunks relevant to the query.
        
        Args:
            chunks: List of document chunks to summarize
            query_info: Query information from QueryProcessor
            images: List of images related to retrieved chunks
            tables: List of tables related to retrieved chunks
            max_length: Target maximum length of summary in words
            
        Returns:
            Dictionary with summary text and references to images/tables
        """
        if not chunks:
            return {
                "summary": "No relevant information found.",
                "referenced_images": [],
                "referenced_tables": []
            }
            
        try:
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Process and clean the texts
            processed_texts = self._preprocess_texts(texts)
            if not processed_texts:
                return {
                    "summary": "No relevant information found after processing.",
                    "referenced_images": [],
                    "referenced_tables": []
                }
                
            # Extract query intent and key concepts from query_info
            query_concepts = self._extract_key_concepts_from_query(query_info)
            
            # Extract information units (sentences or paragraphs)
            information_units = self._extract_information_units(processed_texts, chunks)
            
            # Score information units by relevance to query
            scored_units = self._score_by_query_relevance(information_units, query_concepts, query_info["original_query"])
            
            # Remove redundant information
            unique_units = self._remove_redundancies(scored_units)
            
            # Select top units within target length
            selected_units = self._select_within_length(unique_units, max_length)
            
            # Organize information logically
            organized_units = self._organize_information(selected_units, query_concepts)
            
            # Find relevant images and tables
            referenced_images = []
            referenced_tables = []
            
            if images or tables:
                referenced_images, referenced_tables = self._find_relevant_media(
                    organized_units, query_info, images, tables
                )
            
            # Generate the final summary
            summary = self._generate_coherent_text(organized_units, query_info["original_query"], query_concepts)
            
            return {
                "summary": summary,
                "referenced_images": referenced_images,
                "referenced_tables": referenced_tables
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fall back to traditional summarization if available
            if sumy_available:
                fallback_summary = self._generate_fallback_summary(texts, query_info["original_query"])
                return {
                    "summary": fallback_summary,
                    "referenced_images": [],
                    "referenced_tables": []
                }
            else:
                return {
                    "summary": f"An error occurred while generating the summary. Please try again with a different query.",
                    "referenced_images": [],
                    "referenced_tables": []
                }
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Clean and normalize the input texts."""
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                continue
                
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove excessive punctuation
            text = re.sub(r'([.!?])\1+', r'\1', text)
            
            # Remove content in brackets if it looks like citations or references
            text = re.sub(r'\[\d+\]', '', text)
            
            processed_texts.append(text.strip())
            
        return processed_texts
    
    def _extract_key_concepts_from_query(self, query_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key concepts from the query info."""
        concepts = {
            'keywords': query_info.get('keywords', []),
            'entities': query_info.get('entities', []),
            'query_type': query_info.get('query_type', 'unknown'),
            'question_type': query_info.get('question_type'),
            'original_query': query_info.get('original_query', ''),
            'has_visual_reference': query_info.get('has_visual_reference', False),
            'has_table_reference': query_info.get('has_table_reference', False)
        }
        
        return concepts
    
    def _extract_information_units(self, texts: List[str], chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Extract meaningful information units from texts."""
        # Extract units along with their source chunk information
        all_units = []
        
        for i, text in enumerate(texts):
            # Get the corresponding chunk
            chunk = chunks[i] if i < len(chunks) else None
            
            # If chunk is a title or heading, keep it as a unit
            if chunk and (chunk.is_title or chunk.is_heading):
                all_units.append({
                    'text': text,
                    'chunk': chunk,
                    'is_title': chunk.is_title,
                    'is_heading': chunk.is_heading
                })
                continue
                
            # For longer texts, split by sentences
            if len(text) > 100:
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    # Skip very short sentences
                    if len(sent.split()) < 5:
                        continue
                        
                    all_units.append({
                        'text': sent,
                        'chunk': chunk,
                        'is_title': False,
                        'is_heading': False
                    })
            else:
                # For shorter texts, use as is
                all_units.append({
                    'text': text,
                    'chunk': chunk,
                    'is_title': False,
                    'is_heading': False
                })
                
        # Filter out units with little information
        filtered_units = []
        for unit in all_units:
            # Always keep titles and headings
            if unit['is_title'] or unit['is_heading']:
                filtered_units.append(unit)
                continue
                
            # Skip units that are mostly stopwords
            tokens = unit['text'].lower().split()
            if not tokens:
                continue
                
            non_stop_ratio = sum(1 for t in tokens if t not in self.stop_words) / len(tokens)
            if non_stop_ratio < 0.4:
                continue
                
            filtered_units.append(unit)
            
        return filtered_units
    
    def _score_by_query_relevance(self, 
                                  units: List[Dict[str, Any]], 
                                  query_concepts: Dict[str, Any],
                                  original_query: str) -> List[Tuple[Dict[str, Any], float]]:
        """Score information units by relevance to the query concepts."""
        if not units:
            return []
            
        scored_units = []
        
        # Convert query keywords to a space-separated string for comparison
        query_text = ' '.join(query_concepts['keywords']) if query_concepts['keywords'] else original_query
        
        # Use the semantic processor to calculate relevance
        for unit in units:
            # Base score on semantic similarity
            similarity = self.semantic_processor.calculate_similarity(query_text, unit['text'])
            
            # Apply additional scoring based on query concepts
            bonus = 0
            
            # Boost titles and headings
            if unit['is_title']:
                bonus += 0.3
            elif unit['is_heading']:
                bonus += 0.2
            
            # Check for exact entity matches
            for entity, _ in query_concepts['entities']:
                if entity.lower() in unit['text'].lower():
                    bonus += 0.2
            
            # Check if unit contains information related to question type
            if query_concepts['question_type'] == 'definition' and any(phrase in unit['text'].lower() 
                                                                      for phrase in ['is a', 'refers to', 'defined as']):
                bonus += 0.15
            elif query_concepts['question_type'] == 'process' and any(phrase in unit['text'].lower() 
                                                                     for phrase in ['steps', 'process', 'how to']):
                bonus += 0.15
                
            # Boost units with images/tables if query has visual/table references
            if query_concepts.get('has_visual_reference', False) and unit['chunk'] and unit['chunk'].images:
                bonus += 0.25
                
            if query_concepts.get('has_table_reference', False) and unit['chunk'] and unit['chunk'].tables:
                bonus += 0.25
                
            # Apply bonuses (capped at 1.0)
            final_score = min(1.0, similarity + bonus)
            scored_units.append((unit, final_score))
                
        # Sort by score in descending order
        scored_units.sort(key=lambda x: x[1], reverse=True)
        return scored_units
    
    def _remove_redundancies(self, scored_units: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Remove redundant information to avoid repetition."""
        if not scored_units:
            return []
            
        # Extract units and scores
        units = [item[0]['text'] for item in scored_units]
        original_items = [item for item in scored_units]
        
        # Calculate similarity matrix between all units
        similarity_matrix = self.semantic_processor.calculate_similarity_matrix(units)
        
        # Initialize list to keep track of which units to keep
        keep_mask = np.ones(len(units), dtype=bool)
        
        # Iterate through units by score (already sorted)
        for i in range(len(units)):
            if not keep_mask[i]:
                continue  # Skip if already marked for removal
                
            # Check this unit against all subsequent (lower-scored) units
            for j in range(i + 1, len(units)):
                if not keep_mask[j]:
                    continue  # Skip if already marked for removal
                    
                # If units are too similar, remove the lower-scored one
                if similarity_matrix[i, j] > 0.7:  # Threshold for similarity
                    keep_mask[j] = False
                    
        # Create filtered list of non-redundant units with their scores
        unique_units = [original_items[i] for i in range(len(units)) if keep_mask[i]]
        return unique_units
    
    def _select_within_length(self, scored_units: List[Tuple[Dict[str, Any], float]], max_length: int) -> List[Tuple[Dict[str, Any], float]]:
        """Select top units that fit within the target length."""
        if not scored_units:
            return []
            
        current_length = 0
        selected_units = []
        
        for unit, score in scored_units:
            unit_length = len(unit['text'].split())
            if current_length + unit_length <= max_length:
                selected_units.append((unit, score))
                current_length += unit_length
            
            if current_length >= max_length:
                break
                
        return selected_units
    
    def _organize_information(self, scored_units: List[Tuple[Dict[str, Any], float]], query_concepts: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Organize selected information into a logical structure."""
        if not scored_units:
            return []
            
        # Determine organization strategy based on query
        if query_concepts['question_type'] == 'definition':
            # For definition queries, put definitive statements first
            definitions = []
            elaborations = []
            examples = []
            
            for unit, score in scored_units:
                text = unit['text'].lower()
                if unit['is_title'] or unit['is_heading']:
                    # Titles and headings go first
                    definitions.append((unit, score))
                elif any(phrase in text for phrase in ['is a', 'refers to', 'defined as', 'meaning of']):
                    definitions.append((unit, score))
                elif 'example' in text or 'instance' in text or 'such as' in text:
                    examples.append((unit, score))
                else:
                    elaborations.append((unit, score))
                    
            # Combine in logical order: definition, elaboration, examples
            return definitions + elaborations + examples
            
        elif query_concepts['question_type'] == 'process':
            # Try to organize steps in a process
            # Look for indicators of sequential steps
            units_with_markers = []
            
            for unit, score in scored_units:
                text = unit['text'].lower()
                # Look for step indicators
                has_step = bool(re.search(r'(?:^|\W)(?:step\s*\d|first|second|third|next|then|finally)(?:\W|$)', text))
                priority = 2 if unit['is_title'] else 1 if unit['is_heading'] else 0
                units_with_markers.append((unit, score, has_step, priority))
            
            # Sort: first by priority, then by step marker, then by score
            ordered_units = sorted(units_with_markers, key=lambda x: (-x[3], -x[2], -x[1]))
            return [(unit, score) for unit, score, _, _ in ordered_units]
            
        elif query_concepts['query_type'] == 'explanation':
            # For explanations, start with overview then details
            overview = []
            details = []
            
            for unit, score in scored_units:
                if unit['is_title'] or unit['is_heading']:
                    overview.append((unit, score))
                elif len(unit['text'].split()) < 20:  # Shorter sentences often give overviews
                    overview.append((unit, score))
                else:
                    details.append((unit, score))
                    
            return overview + details
            
        else:
            # For other types, start with titles/headings, then highest relevance units
            titles = []
            others = []
            
            for unit, score in scored_units:
                if unit['is_title'] or unit['is_heading']:
                    titles.append((unit, score))
                else:
                    others.append((unit, score))
                    
            # Sort non-titles by score
            others.sort(key=lambda x: x[1], reverse=True)
            
            return titles + others
    
    def _find_relevant_media(self, 
                            organized_units: List[Tuple[Dict[str, Any], float]], 
                            query_info: Dict[str, Any],
                            images: List[ImageData],
                            tables: List[TableData]) -> Tuple[List[str], List[str]]:
        """Find relevant images and tables based on query and selected units."""
        referenced_images = []
        referenced_tables = []
        
        # Check if we have images and tables
        if not images and not tables:
            return [], []
        
        # Extract all chunks used in the summary
        used_chunks = [unit['chunk'] for unit, _ in organized_units if unit['chunk']]
        
        # First pass: collect media referenced in chunks used in summary
        for chunk in used_chunks:
            if chunk.images:
                for img_id in chunk.images:
                    if img_id not in referenced_images:
                        referenced_images.append(img_id)
                        
            if chunk.tables:
                for table_id in chunk.tables:
                    if table_id not in referenced_tables:
                        referenced_tables.append(table_id)
        
        # Second pass: check if the query specifically asks for visual content
        if query_info.get('has_visual_reference', False) and not referenced_images and images:
            # Add most relevant images based on image type
            relevant_image_types = []
            
            # Determine relevant image types from query
            query_text = query_info['original_query'].lower()
            if any(term in query_text for term in ['chart', 'graph', 'plot']):
                relevant_image_types.append('chart')
            if any(term in query_text for term in ['diagram', 'flow', 'architecture']):
                relevant_image_types.append('diagram')
            if any(term in query_text for term in ['photo', 'picture', 'image']):
                relevant_image_types.append('photo')
                
            # If no specific type mentioned, consider all types relevant
            if not relevant_image_types:
                relevant_image_types = ['chart', 'diagram', 'photo', 'table', 'other']
                
            # Find images of relevant types
            for img in images:
                if img.image_type in relevant_image_types and img.image_id not in referenced_images:
                    referenced_images.append(img.image_id)
                    # Limit to top 3 most relevant images
                    if len(referenced_images) >= 3:
                        break
        
        # Third pass: check if the query specifically asks for tables
        if query_info.get('has_table_reference', False) and not referenced_tables and tables:
            # Add up to 2 tables
            for table in tables:
                if table.table_id not in referenced_tables:
                    referenced_tables.append(table.table_id)
     


def _find_relevant_media(self, 
                            organized_units: List[Tuple[Dict[str, Any], float]], 
                            query_info: Dict[str, Any],
                            images: List[ImageData],
                            tables: List[TableData]) -> Tuple[List[str], List[str]]:
        """Find relevant images and tables based on query and selected units."""
        referenced_images = []
        referenced_tables = []
        
        # Check if we have images and tables
        if not images and not tables:
            return [], []
        
        # Extract all chunks used in the summary
        used_chunks = [unit['chunk'] for unit, _ in organized_units if unit['chunk']]
        
        # First pass: collect media referenced in chunks used in summary
        for chunk in used_chunks:
            if chunk.images:
                for img_id in chunk.images:
                    if img_id not in referenced_images:
                        referenced_images.append(img_id)
                        
            if chunk.tables:
                for table_id in chunk.tables:
                    if table_id not in referenced_tables:
                        referenced_tables.append(table_id)
        
        # Second pass: check if the query specifically asks for visual content
        if query_info.get('has_visual_reference', False) and not referenced_images and images:
            # Add most relevant images based on image type
            relevant_image_types = []
            
            # Determine relevant image types from query
            query_text = query_info['original_query'].lower()
            if any(term in query_text for term in ['chart', 'graph', 'plot']):
                relevant_image_types.append('chart')
            if any(term in query_text for term in ['diagram', 'flow', 'architecture']):
                relevant_image_types.append('diagram')
            if any(term in query_text for term in ['photo', 'picture', 'image']):
                relevant_image_types.append('photo')
                
            # If no specific type mentioned, consider all types relevant
            if not relevant_image_types:
                relevant_image_types = ['chart', 'diagram', 'photo', 'table', 'other']
                
            # Find images of relevant types
            for img in images:
                if img.image_type in relevant_image_types and img.image_id not in referenced_images:
                    referenced_images.append(img.image_id)
                    # Limit to top 3 most relevant images
                    if len(referenced_images) >= 3:
                        break
        
        # Third pass: check if the query specifically asks for tables
        if query_info.get('has_table_reference', False) and not referenced_tables and tables:
            # Add up to 2 tables
            for table in tables:
                if table.table_id not in referenced_tables:
                    referenced_tables.append(table.table_id)
                    # Limit to top 2 tables
                    if len(referenced_tables) >= 2:
                        break
                        
        return referenced_images, referenced_tables
    
    def _generate_coherent_text(self, 
                                organized_units: List[Tuple[Dict[str, Any], float]], 
                                query: str,
                                query_concepts: Dict[str, Any]) -> str:
        """Generate the final coherent summary text."""
        if not organized_units:
            return "No relevant information was found to answer your query."
            
        # Extract units and drop scores
        units = [unit['text'] for unit, _ in organized_units]
        
        # Start with an introduction based on query type
        introduction = self._create_introduction(query, query_concepts)
        
        # Combine units into paragraphs
        paragraphs = self._create_paragraphs(organized_units)
        
        # Add a conclusion
        conclusion = self._create_conclusion(query, query_concepts)
        
        # Combine all parts
        full_text = introduction + "\n\n"
        full_text += "\n\n".join(paragraphs)
        
        if conclusion:
            full_text += "\n\n" + conclusion
            
        return full_text
    
    def _create_introduction(self, query: str, query_concepts: Dict[str, Any]) -> str:
        """Create an introductory sentence based on the query."""
        # Extract main subject from entities if available
        subject = None
        if query_concepts['entities']:
            subject = query_concepts['entities'][0][0]
        
        # If no entities, check for keywords
        if not subject and query_concepts['keywords']:
            subject = query_concepts['keywords'][0].capitalize()
        
        # Generate introduction based on query type
        if query_concepts['query_type'] == 'question':
            if query_concepts['question_type'] == 'definition':
                if subject:
                    return f"Here's information about {subject}:"
                else:
                    return "Here's the definition you requested:"
            elif query_concepts['question_type'] == 'process':
                return "Here's how the process works:"
            elif query_concepts['question_type'] == 'reason':
                return "Here's the explanation for your question:"
            else:
                return "Here's information addressing your question:"
        elif query_concepts['query_type'] == 'explanation':
            return "Here's an explanation of the topic:"
        elif query_concepts['query_type'] == 'search':
            if subject:
                return f"Here's what I found about {subject}:"
            else:
                return "Here are the search results:"
        else:
            if subject:
                return f"Here's a summary of information about {subject}:"
            else:
                return "Here's a summary of the relevant information:"
    
    def _create_paragraphs(self, scored_units: List[Tuple[Dict[str, Any], float]]) -> List[str]:
        """Combine information units into coherent paragraphs."""
        if not scored_units:
            return []
            
        # Group by chunk to maintain document coherence when possible
        chunk_groups = {}
        for unit, _ in scored_units:
            chunk_id = unit['chunk'].doc_id if unit['chunk'] else "unknown"
            if chunk_id not in chunk_groups:
                chunk_groups[chunk_id] = []
            chunk_groups[chunk_id].append(unit)
        
        paragraphs = []
        
        # Process titles and headings first
        for chunk_id, units in chunk_groups.items():
            titles = [unit for unit in units if unit['is_title']]
            headings = [unit for unit in units if unit['is_heading']]
            
            # Add titles
            for title in titles:
                paragraphs.append(title['text'])
            
            # Add headings
            for heading in headings:
                paragraphs.append(heading['text'])
        
        # Process regular content
        for chunk_id, units in chunk_groups.items():
            regular_units = [unit for unit in units if not unit['is_title'] and not unit['is_heading']]
            
            # Skip if no regular units
            if not regular_units:
                continue
                
            # If just 1-2 units, add as separate paragraphs
            if len(regular_units) <= 2:
                for unit in regular_units:
                    paragraphs.append(unit['text'])
                continue
                
            # Otherwise, try to combine related units into coherent paragraphs
            current_paragraph = []
            
            for i, unit in enumerate(regular_units):
                if i == 0:
                    current_paragraph.append(unit['text'])
                    continue
                    
                # Check similarity with last unit in current paragraph
                last_text = current_paragraph[-1]
                
                similarity = self.semantic_processor.calculate_similarity(last_text, unit['text'])
                
                if similarity > 0.5 and len(current_paragraph) < 3:
                    # Add to current paragraph if related and paragraph not too long yet
                    current_paragraph.append(unit['text'])
                else:
                    # Start a new paragraph
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = [unit['text']]
            
            # Add the last paragraph
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                
        return paragraphs
    
    def _create_conclusion(self, query: str, query_concepts: Dict[str, Any]) -> str:
        """Create a concluding sentence if appropriate."""
        # Determine if conclusion is needed based on query type
        if query_concepts['query_type'] == 'question' and query_concepts['question_type'] in ['reason', 'process']:
            return "This summary addresses the key points based on the available information."
        elif query_concepts['query_type'] == 'explanation':
            return "This explanation covers the main aspects of the topic as requested."
        elif len(query.split()) > 10:  # Longer, more complex queries often benefit from conclusion
            return "The information above represents the most relevant content found in the knowledge base."
        
        return ""  # No conclusion for simple factual queries
        
    def _generate_fallback_summary(self, texts: List[str], query: str) -> str:
        """Generate a summary using traditional methods as fallback."""
        if not sumy_available or not texts:
            return "Could not generate a summary with the available information."
        
        # Combine texts
        combined_text = "\n\n".join(texts)
        
        try:
            # Parse the text
            parser = PlaintextParser.from_string(combined_text, Tokenizer(self.language))
            
            # Use LexRank for extractive summarization
            summary_sentences = [str(s) for s in self.lexrank(parser.document, 10)]
            
            if not summary_sentences:
                return "No relevant information could be extracted."
                
            # Format into a coherent text
            intro = "Here is a summary of the relevant information:"
            body = " ".join(summary_sentences)
            
            return f"{intro}\n\n{body}"
            
        except Exception as e:
            logger.error(f"Error generating fallback summary: {str(e)}")
            return "An error occurred while generating the summary."

class EnhancedPDFReportGenerator:
    """Generates enhanced PDF reports with professional formatting and visual elements."""
    
    def __init__(self, assets_dir: str = None, branding: Dict[str, Any] = None):
        """Initialize the PDF report generator.
        
        Args:
            assets_dir: Directory containing assets like images
            branding: Optional branding information (colors, logo, etc.)
        """
        self.assets_dir = assets_dir or "assets"
        
        # Set up default branding
        self.branding = {
            "primary_color": colors.HexColor("#2C3E50"),  # Dark blue
            "secondary_color": colors.HexColor("#3498DB"),  # Light blue
            "accent_color": colors.HexColor("#E74C3C"),  # Red
            "font_name": "Helvetica",
            "logo_path": None,
            "footer_text": "Generated by Enhanced RAG System"
        }
        
        # Update with custom branding if provided
        if branding:
            self.branding.update(branding)
        
        # Create styles for the PDF
        self.styles = self._create_styles()
    
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create styles for the PDF document."""
        styles = getSampleStyleSheet()
        
        # Update existing styles
        styles["Title"].fontName = self.branding["font_name"] + "-Bold"
        styles["Title"].fontSize = 20
        styles["Title"].leading = 24
        styles["Title"].textColor = self.branding["primary_color"]
        styles["Title"].alignment = TA_CENTER
        styles["Title"].spaceAfter = 12
        
        styles["Heading1"].fontName = self.branding["font_name"] + "-Bold"
        styles["Heading1"].fontSize = 16
        styles["Heading1"].leading = 20
        styles["Heading1"].textColor = self.branding["primary_color"]
        styles["Heading1"].spaceAfter = 10
        
        styles["Heading2"].fontName = self.branding["font_name"] + "-Bold"
        styles["Heading2"].fontSize = 14
        styles["Heading2"].leading = 18
        styles["Heading2"].textColor = self.branding["secondary_color"]
        styles["Heading2"].spaceAfter = 8
        
        styles["Normal"].fontName = self.branding["font_name"]
        styles["Normal"].fontSize = 11
        styles["Normal"].leading = 14
        
        # Add custom styles
        styles.add(ParagraphStyle(
            name='Body',
            fontName=self.branding["font_name"],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=8
        ))
        
        styles.add(ParagraphStyle(
            name='Caption',
            fontName=self.branding["font_name"] + "-Italic",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.dark_gray
        ))
        
        styles.add(ParagraphStyle(
            name='TableHeader',
            fontName=self.branding["font_name"] + "-Bold",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.whitesmoke,
            backColor=self.branding["secondary_color"]
        ))
        
        styles.add(ParagraphStyle(
            name='Footer',
            fontName=self.branding["font_name"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.gray
        ))
        
        styles.add(ParagraphStyle(
            name='ListItem',
            fontName=self.branding["font_name"],
            fontSize=11,
            leading=14,
            leftIndent=20,
            bulletIndent=10
        ))
        
        styles.add(ParagraphStyle(
            name='Note',
            fontName=self.branding["font_name"] + "-Italic",
            fontSize=10,
            leading=12,
            alignment=TA_JUSTIFY,
            textColor=colors.darkslategray
        ))
        
        return styles
    
    def generate_pdf_report(self, 
                           query: str, 
                           response: Dict[str, Any], 
                           retrieved_chunks: List[DocumentChunk],
                           images: Dict[str, ImageData] = None,
                           tables: Dict[str, TableData] = None,
                           document_metadata: Dict[str, DocumentMetadata] = None,
                           output_path: str = None) -> str:
        """Generate a PDF report of the response.
        
        Args:
            query: The original query
            response: Response dict with summary and referenced media
            retrieved_chunks: List of retrieved chunks
            images: Dictionary of image data objects
            tables: Dictionary of table data objects
            document_metadata: Dictionary of document metadata
            output_path: Optional path for the generated PDF
            
        Returns:
            Path to the generated PDF
        """
        # Create output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"response_{timestamp}.pdf"
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create elements for the PDF
            elements = []
            
            # Add logo if available
            if self.branding["logo_path"] and os.path.exists(self.branding["logo_path"]):
                logo = RLImage(self.branding["logo_path"], width=1.5*inch, height=0.5*inch)
                elements.append(logo)
                elements.append(Spacer(1, 12))
            
            # Add title
            title_text = f"Information for query: {query}"
            elements.append(Paragraph(title_text, self.styles["Title"]))
            elements.append(Spacer(1, 20))
            
            # Add generation time
            time_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elements.append(Paragraph(time_text, self.styles["Footer"]))
            elements.append(Spacer(1, 20))
            
            # Add summary
            elements.append(Paragraph("Summary", self.styles["Heading1"]))
            
            # Process summary paragraphs
            summary_paragraphs = response["summary"].split('\n\n')
            for paragraph in summary_paragraphs:
                if not paragraph.strip():
                    continue
                    
                if paragraph.startswith('# '):
                    # Main heading
                    heading_text = paragraph[2:].strip()
                    elements.append(Paragraph(heading_text, self.styles["Heading1"]))
                    elements.append(Spacer(1, 8))
                elif paragraph.startswith('## '):
                    # Subheading
                    subheading_text = paragraph[3:].strip()
                    elements.append(Paragraph(subheading_text, self.styles["Heading2"]))
                    elements.append(Spacer(1, 6))
                elif paragraph.startswith('- '):
                    # List item
                    list_text = paragraph[2:].strip()
                    elements.append(Paragraph("• " + list_text, self.styles["ListItem"]))
                    elements.append(Spacer(1, 4))
                else:
                    # Regular paragraph
                    elements.append(Paragraph(paragraph, self.styles["Body"]))
                    elements.append(Spacer(1, 8))
            
            # Add referenced images if any
            if response.get("referenced_images") and images:
                elements.append(PageBreak())
                elements.append(Paragraph("Referenced Images", self.styles["Heading1"]))
                elements.append(Spacer(1, 12))
                
                for img_id in response["referenced_images"]:
                    if img_id in images:
                        img_data = images[img_id]
                        
                        # Add image caption
                        if img_data.caption:
                            elements.append(Paragraph(f"Figure: {img_data.caption}", self.styles["Heading2"]))
                        else:
                            elements.append(Paragraph(f"Figure from page {img_data.page_num}", self.styles["Heading2"]))
                        
                        elements.append(Spacer(1, 6))
                        
                        # Save image to temp file and add to PDF
                        img_path = img_data.save_to_file(temp_dir)
                        
                        if img_path and os.path.exists(img_path):
                            # Determine image size (fi




class EnhancedPDFReportGenerator:
    """Generates enhanced PDF reports with professional formatting and visual elements."""
    
    def __init__(self, assets_dir: str = None, branding: Dict[str, Any] = None):
        """Initialize the PDF report generator.
        
        Args:
            assets_dir: Directory containing assets like images
            branding: Optional branding information (colors, logo, etc.)
        """
        self.assets_dir = assets_dir or "assets"
        
        # Set up default branding
        self.branding = {
            "primary_color": colors.HexColor("#2C3E50"),  # Dark blue
            "secondary_color": colors.HexColor("#3498DB"),  # Light blue
            "accent_color": colors.HexColor("#E74C3C"),  # Red
            "font_name": "Helvetica",
            "logo_path": None,
            "footer_text": "Generated by Enhanced RAG System"
        }
        
        # Update with custom branding if provided
        if branding:
            self.branding.update(branding)
        
        # Create styles for the PDF
        self.styles = self._create_styles()
    
    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create styles for the PDF document."""
        styles = getSampleStyleSheet()
        
        # Update existing styles
        styles["Title"].fontName = self.branding["font_name"] + "-Bold"
        styles["Title"].fontSize = 20
        styles["Title"].leading = 24
        styles["Title"].textColor = self.branding["primary_color"]
        styles["Title"].alignment = TA_CENTER
        styles["Title"].spaceAfter = 12
        
        styles["Heading1"].fontName = self.branding["font_name"] + "-Bold"
        styles["Heading1"].fontSize = 16
        styles["Heading1"].leading = 20
        styles["Heading1"].textColor = self.branding["primary_color"]
        styles["Heading1"].spaceAfter = 10
        
        styles["Heading2"].fontName = self.branding["font_name"] + "-Bold"
        styles["Heading2"].fontSize = 14
        styles["Heading2"].leading = 18
        styles["Heading2"].textColor = self.branding["secondary_color"]
        styles["Heading2"].spaceAfter = 8
        
        styles["Normal"].fontName = self.branding["font_name"]
        styles["Normal"].fontSize = 11
        styles["Normal"].leading = 14
        
        # Add custom styles
        styles.add(ParagraphStyle(
            name='Body',
            fontName=self.branding["font_name"],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=8
        ))
        
        styles.add(ParagraphStyle(
            name='Caption',
            fontName=self.branding["font_name"] + "-Italic",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.dark_gray
        ))
        
        styles.add(ParagraphStyle(
            name='TableHeader',
            fontName=self.branding["font_name"] + "-Bold",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.whitesmoke,
            backColor=self.branding["secondary_color"]
        ))
        
        styles.add(ParagraphStyle(
            name='Footer',
            fontName=self.branding["font_name"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.gray
        ))
        
        styles.add(ParagraphStyle(
            name='ListItem',
            fontName=self.branding["font_name"],
            fontSize=11,
            leading=14,
            leftIndent=20,
            bulletIndent=10
        ))
        
        styles.add(ParagraphStyle(
            name='Note',
            fontName=self.branding["font_name"] + "-Italic",
            fontSize=10,
            leading=12,
            alignment=TA_JUSTIFY,
            textColor=colors.darkslategray
        ))
        
        return styles
    
    def generate_pdf_report(self, 
                           query: str, 
                           response: Dict[str, Any], 
                           retrieved_chunks: List[DocumentChunk],
                           images: Dict[str, ImageData] = None,
                           tables: Dict[str, TableData] = None,
                           document_metadata: Dict[str, DocumentMetadata] = None,
                           output_path: str = None) -> str:
        """Generate a PDF report of the response.
        
        Args:
            query: The original query
            response: Response dict with summary and referenced media
            retrieved_chunks: List of retrieved chunks
            images: Dictionary of image data objects
            tables: Dictionary of table data objects
            document_metadata: Dictionary of document metadata
            output_path: Optional path for the generated PDF
            
        Returns:
            Path to the generated PDF
        """
        # Create output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"response_{timestamp}.pdf"
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Create elements for the PDF
            elements = []
            
            # Add logo if available
            if self.branding["logo_path"] and os.path.exists(self.branding["logo_path"]):
                logo = RLImage(self.branding["logo_path"], width=1.5*inch, height=0.5*inch)
                elements.append(logo)
                elements.append(Spacer(1, 12))
            
            # Add title
            title_text = f"Information for query: {query}"
            elements.append(Paragraph(title_text, self.styles["Title"]))
            elements.append(Spacer(1, 20))
            
            # Add generation time
            time_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            elements.append(Paragraph(time_text, self.styles["Footer"]))
            elements.append(Spacer(1, 20))
            
            # Add summary
            elements.append(Paragraph("Summary", self.styles["Heading1"]))
            
            # Process summary paragraphs
            summary_paragraphs = response["summary"].split('\n\n')
            for paragraph in summary_paragraphs:
                if not paragraph.strip():
                    continue
                    
                if paragraph.startswith('# '):
                    # Main heading
                    heading_text = paragraph[2:].strip()
                    elements.append(Paragraph(heading_text, self.styles["Heading1"]))
                    elements.append(Spacer(1, 8))
                elif paragraph.startswith('## '):
                    # Subheading
                    subheading_text = paragraph[3:].strip()
                    elements.append(Paragraph(subheading_text, self.styles["Heading2"]))
                    elements.append(Spacer(1, 6))
                elif paragraph.startswith('- '):
                    # List item
                    list_text = paragraph[2:].strip()
                    elements.append(Paragraph("• " + list_text, self.styles["ListItem"]))
                    elements.append(Spacer(1, 4))
                else:
                    # Regular paragraph
                    elements.append(Paragraph(paragraph, self.styles["Body"]))
                    elements.append(Spacer(1, 8))
            
            # Add referenced images if any
            if response.get("referenced_images") and images:
                elements.append(PageBreak())
                elements.append(Paragraph("Referenced Images", self.styles["Heading1"]))
                elements.append(Spacer(1, 12))
                
                for img_id in response["referenced_images"]:
                    if img_id in images:
                        img_data = images[img_id]
                        
                        # Add image caption
                        if img_data.caption:
                            elements.append(Paragraph(f"Figure: {img_data.caption}", self.styles["Heading2"]))
                        else:
                            elements.append(Paragraph(f"Figure from page {img_data.page_num}", self.styles["Heading2"]))
                        
                        elements.append(Spacer(1, 6))
                        
                        # Save image to temp file and add to PDF
                        img_path = img_data.save_to_file(temp_dir)
                        
                        if img_path and os.path.exists(img_path):
                            # Determine image size (fit within page width)
                            max_width = doc.width * 0.9
                            img = RLImage(img_path)
                            
                            # Scale image if necessary
                            if img.drawWidth > max_width:
                                height_ratio = img.drawHeight / img.drawWidth
                                img.drawWidth = max_width
                                img.drawHeight = max_width * height_ratio
                            
                            elements.append(img)
                            elements.append(Spacer(1, 8))
                            
                            # Add OCR text if available
                            if img_data.ocr_text:
                                elements.append(Paragraph("Text extracted from image:", self.styles["Note"]))
                                elements.append(Paragraph(img_data.ocr_text, self.styles["Body"]))
                                elements.append(Spacer(1, 12))
            
            # Add referenced tables if any
            if response.get("referenced_tables") and tables:
                elements.append(PageBreak())
                elements.append(Paragraph("Referenced Tables", self.styles["Heading1"]))
                elements.append(Spacer(1, 12))
                
                for table_id in response["referenced_tables"]:
                    if table_id in tables:
                        table_data = tables[table_id]
                        
                        # Add table caption
                        if table_data.caption:
                            elements.append(Paragraph(f"Table: {table_data.caption}", self.styles["Heading2"]))
                        else:
                            elements.append(Paragraph(f"Table from page {table_data.page_num}", self.styles["Heading2"]))
                        
                        elements.append(Spacer(1, 6))
                        
                        # Convert pandas DataFrame to ReportLab table
                        df = table_data.table_data
                        
                        # Prepare data for ReportLab table
                        rl_data = [df.columns.tolist()]  # Header row
                        rl_data.extend(df.values.tolist())  # Data rows
                        
                        # Convert data to strings and handle NaN
                        for i in range(len(rl_data)):
                            rl_data[i] = [str(cell) if cell is not None and cell == cell else "" for cell in rl_data[i]]
                        
                        # Create table
                        table = Table(rl_data)
                        
                        # Style the table
                        table_style = TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), self.branding["secondary_color"]),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), self.branding["font_name"] + '-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 10),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 1), (-1, -1), self.branding["font_name"]),
                            ('FONTSIZE', (0, 1), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ])
                        
                        # Add alternating row colors
                        for i in range(1, len(rl_data), 2):
                            table_style.add('BACKGROUND', (0, i), (-1, i), colors.whitesmoke)
                        
                        table.setStyle(table_style)
                        
                        # Make sure table fits in the page
                        table_width = min(doc.width * 0.95, table._width)
                        table_scale = table_width / table._width if table._width > 0 else 1
                        
                        if table_scale < 1:
                            table = Table(rl_data, colWidths=[w * table_scale for w in table._colWidths])
                            table.setStyle(table_style)
                            
                        elements.append(table)
                        elements.append(Spacer(1, 12))
            
            # Add sources
            elements.append(PageBreak())
            elements.append(Paragraph("Sources", self.styles["Heading1"]))
            elements.append(Spacer(1, 10))
            
            # Group sources by document
            doc_sources = {}
            for chunk in retrieved_chunks:
                if chunk.doc_id not in doc_sources:
                    doc_sources[chunk.doc_id] = []
                doc_sources[chunk.doc_id].append(chunk)
            
            # Add source information
            for doc_id, chunks in doc_sources.items():
                # Add document title and metadata if available
                if document_metadata and doc_id in document_metadata:
                    metadata = document_metadata[doc_id]
                    doc_title = metadata.title or doc_id
                    elements.append(Paragraph(doc_title, self.styles["Heading2"]))
                    
                    # Add metadata table
                    metadata_data = [
                        ["Property", "Value"],
                        ["File name", doc_id],
                        ["Pages", str(metadata.num_pages)],
                        ["Author", metadata.author],
                        ["Creation date", metadata.creation_date]
                    ]
                    
                    metadata_table = Table(metadata_data, colWidths=[1.5*inch, 4*inch])
                    metadata_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (1, 0), self.branding["secondary_color"]),
                        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (1, 0), self.branding["font_name"] + '-Bold'),
                        ('FONTSIZE', (0, 0), (1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (1, 0), 6),
                        ('BACKGROUND', (0, 1), (0, -1), colors.lavender),
                        ('BACKGROUND', (1, 1), (1, -1), colors.white),
                        ('GRID', (0, 0), (1, -1), 0.5, colors.lightgrey),
                        ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
                    ]))
                    
                    elements.append(metadata_table)
                else:
                    elements.append(Paragraph(doc_id, self.styles["Heading2"]))
                
                elements.append(Spacer(1, 8))
                
                # List relevant chunks with page numbers
                pages_used = sorted(set(chunk.page_num for chunk in chunks))
                if pages_used:
                    page_text = "Pages referenced: " + ", ".join(str(p) for p in pages_used)
                    elements.append(Paragraph(page_text, self.styles["Body"]))
                    elements.append(Spacer(1, 12))
            
            # Add footer
            def add_footer(canvas, doc):
                canvas.saveState()
                canvas.setFont(self.branding["font_name"], 8)
                canvas.setFillColor(colors.grey)
                canvas.drawString(inch, 0.5 * inch, self.branding["footer_text"])
                canvas.drawString(4 * inch, 0.5 * inch, f"Page {doc.page}")
                canvas.restoreState()
            
            # Build the PDF
            try:
                doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
                logger.info(f"PDF report generated at: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error building PDF: {str(e)}")
                return None





class EvaluationFramework:
    """Framework for evaluating RAG system performance."""
    
    def __init__(self):
        """Initialize the evaluation framework."""
        self.rouge_evaluator = Rouge() if rouge_available else None
    
    def evaluate_response(self, 
                         query: str, 
                         response: Dict[str, Any], 
                         retrieved_chunks: List[DocumentChunk],
                         reference_summary: str = None,
                         relevant_chunks: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG system response.
        
        Args:
            query: The original query
            response: The generated response dict
            retrieved_chunks: The chunks retrieved by the system
            reference_summary: Optional reference summary for comparison
            relevant_chunks: Optional list of chunk IDs known to be relevant
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "query": query,
            "metrics": {}
        }
        
        # Get summary text from response dict
        summary = response.get("summary", "")
        
        # Calculate response quality metrics
        results["metrics"]["response_length"] = len(summary.split())
        
        # Calculate readability metrics
        results["metrics"]["readability"] = self._calculate_readability(summary)
        
        # Calculate multimedia enrichment score
        results["metrics"]["multimedia_score"] = self._calculate_multimedia_score(response)
        
        # Calculate retrieval precision if relevant chunks are provided
        if relevant_chunks and retrieved_chunks:
            retrieved_ids = [f"{chunk.doc_id}_{chunk.chunk_id}" for chunk in retrieved_chunks]
            precision = len(set(retrieved_ids).intersection(relevant_chunks)) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(set(retrieved_ids).intersection(relevant_chunks)) / len(relevant_chunks) if relevant_chunks else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results["metrics"]["retrieval_precision"] = precision
            results["metrics"]["retrieval_recall"] = recall
            results["metrics"]["retrieval_f1"] = f1
        
        # Calculate ROUGE metrics if reference summary is provided
        if reference_summary and self.rouge_evaluator and summary:
            try:
                rouge_scores = self.rouge_evaluator.get_scores(summary, reference_summary)
                
                results["metrics"]["rouge_1_f"] = rouge_scores[0]["rouge-1"]["f"]
                results["metrics"]["rouge_2_f"] = rouge_scores[0]["rouge-2"]["f"]
                results["metrics"]["rouge_l_f"] = rouge_scores[0]["rouge-l"]["f"]
            except Exception as e:
                logger.error(f"Error calculating ROUGE metrics: {str(e)}")
        
        return results
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for a text."""
        # Count sentences, words, and syllables
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if not sentences or not words:
            return {
                "flesch_reading_ease": 0,
                "avg_sentence_length": 0
            }
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Estimate syllables (simplistic approach)
        syllable_count = 0
        for word in words:
            word = word.lower()
            if len(word) <= 3:
                syllable_count += 1
            else:
                # Count vowel sequences as syllables
                vowels = "aeiouy"
                prev_is_vowel = False
                count = 0
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_is_vowel:
                        count += 1
                    prev_is_vowel = is_vowel
                
                # Adjust counts for typical patterns
                if word.endswith('e'):
                    count -= 1
                if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                    count += 1
                if count == 0:
                    count = 1
                    
                syllable_count += count
        
        # Calculate Flesch Reading Ease
        if len(words) > 0 and len(sentences) > 0:
            flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / len(words)))
        else:
            flesch = 0
            
        return {
            "flesch_reading_ease": flesch,
            "avg_sentence_length": avg_sentence_length
        }
    
    def _calculate_multimedia_score(self, response: Dict[str, Any]) -> float:
        """Calculate a score for multimedia enrichment of the response."""
        # Base score
        score = 0
        
        # Check for images
        if response.get("referenced_images"):
            # Add points for each image (diminishing returns)
            num_images = len(response["referenced_images"])
            score += min(0.5, 0.2 * num_images)
        
        # Check for tables
        if response.get("referenced_tables"):
            # Add points for each table (diminishing returns)
            num_tables = len(response["referenced_tables"])
            score += min(0.5, 0.25 * num_tables)
        
        # Check if the response mentions the tables/images
        if response.get("summary"):
            summary = response["summary"].lower()
            
            has_image_reference = any(term in summary for term in ["figure", "image", "picture", "chart", "graph", "diagram"])
            has_table_reference = any(term in summary for term in ["table", "data", "values", "figures", "numbers"])
            
            if has_image_reference and response.get("referenced_images"):
                score += 0.1
                
            if has_table_reference and response.get("referenced_tables"):
                score += 0.1
                
        return min(1.0, score)  # Cap at 1.0
    
    def benchmark_system(self, rag_system, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark the RAG system on a set of test queries.
        
        Args:
            rag_system: The RAG system to evaluate
            test_queries: List of dictionaries with query, relevant_chunks, and optional reference_summary
            
        Returns:
            Dictionary with aggregate benchmark results
        """
        if not test_queries:
            return {"error": "No test queries provided"}
            
        results = {
            "individual_results": [],
            "aggregate_metrics": {},
            "timing": {
                "total_time": 0,
                "avg_time": 0
            }
        }
        
        start_time = time.time()
        
        for test in test_queries:
            query = test["query"]
            
            # Process the query and measure time
            query_start = time.time()
            response = rag_system.process_query(query)
            query_time = time.time() - query_start
            
            # Evaluate the response
            eval_result = self.evaluate_response(
                query=query,
                response=response.get("response", {}),
                retrieved_chunks=response.get("retrieved_chunks", []),
                reference_summary=test.get("reference_summary"),
                relevant_chunks=test.get("relevant_chunks")
            )
            
            # Add timing information
            eval_result["time_taken"] = query_time
            
            # Add to individual results
            results["individual_results"].append(eval_result)
            
        # Calculate aggregate metrics
        results["timing"]["total_time"] = time.time() - start_time
        results["timing"]["avg_time"] = results["timing"]["total_time"] / len(test_queries)
        
        # Calculate average metrics across all queries
        metrics_keys = results["individual_results"][0]["metrics"].keys()
        for key in metrics_keys:
            values = [r["metrics"].get(key, 0) for r in results["individual_results"] if key in r["metrics"]]
            if values:
                results["aggregate_metrics"][key] = sum(values) / len(values)
            
        return results




class MultimodalRAGSystem:
    """Enhanced RAG system with multimodal support and conversation context."""
    
    def __init__(self, knowledge_base_path: str, vector_db_path: str = None, 
                parallel_processing: bool = True, branding: Dict[str, Any] = None,
                auto_process: bool = False):
        """Initialize the RAG system.
        
        Args:
            knowledge_base_path: Path to knowledge base documents
            vector_db_path: Optional custom path for vector DB
            parallel_processing: Whether to use parallel processing for document processing
            branding: Optional branding information for PDF reports
            auto_process: Whether to automatically process documents on initialization
        """
        logger.info(f"Initializing Multimodal RAG system with knowledge base path: {knowledge_base_path}")
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(knowledge_base_path):
            os.makedirs(knowledge_base_path, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_path}")
            
        # Create assets directory
        self.assets_dir = os.path.join(knowledge_base_path, "assets")
        os.makedirs(self.assets_dir, exist_ok=True)
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(knowledge_base_path, vector_db_path, parallel_processing)
        
        # Initialize other components
        self.query_processor = QueryProcessor()
        self.summary_generator = ImprovedSummaryGenerator()
        self.pdf_generator = EnhancedPDFReportGenerator(self.assets_dir, branding)
        self.evaluation = EvaluationFramework()
        
        # Conversation history
        self.conversation_history = []
        self.last_retrieved_chunks = []
        
        # Process documents if auto_process is enabled
        if auto_process:
            self.document_processor.process_all_documents()
    
    def process_query(self, query: str, 
                     consider_history: bool = True, 
                     verbose: bool = False) -> Dict[str, Any]:
        """Process a query and generate a response with conversation context.
        
        Args:
            query: The user's query
            consider_history: Whether to consider conversation history
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary with response information
        """
        logger.info(f"Processing query: {query}")
        
        # Load documents if not already loaded
        if not self.document_processor.document_chunks:
            self.document_processor.process_all_documents()
        
        # Use conversation history if available and enabled
        context_enhanced_query = query
        if consider_history and self.conversation_history:
            # Get context from previous interactions
            context = [item["query"] for item in self.conversation_history[-3:]]
            context.extend([item["response"].get("summary", "") for item in self.conversation_history[-3:]])
            
            # Rewrite query considering context
            context_enhanced_query = self.query_processor.rewrite_query(query, context)
            
            if verbose and context_enhanced_query != query:
                logger.info(f"Rewrote query '{query}' to '{context_enhanced_query}' using conversation context")
        
        # Process the query to understand intent and extract key concepts
        query_info = self.query_processor.process_query(context_enhanced_query)
        
        if verbose:
            logger.info(f"Processed query info: {json.dumps(query_info, indent=2)}")
            
        # Determine query approach (hybrid search for questions, keyword search for simple lookups)
        is_question = query_info["query_type"] in ["question", "explanation"]
        
        # Retrieve relevant chunks
        if is_question:
            # Use expanded query for better retrieval
            retrieved_chunks = self.document_processor.hybrid_search(query_info["expanded_query"], top_k=7)
            chunks = [chunk for chunk, score in retrieved_chunks]
            retrieval_scores = [score for chunk, score in retrieved_chunks]
        else:
            # For keywords, try to find the most specific matches
            keywords = query_info["keywords"]
            # Use the longest keywords for better specificity
            keywords.sort(key=len, reverse=True)
            
            chunks = []
            seen_chunks = set()
            
            # Try exact matching with top keywords
            for keyword in keywords[:3]:  # Try with top 3 longest keywords
                if len(keyword) > 3:  # Only use meaningful keywords
                    matches = self.document_processor.keyword_search(keyword)
                    for doc_id, chunk in matches:
                        chunk_id = f"{doc_id}_{chunk.chunk_id}"
                        if chunk_id not in seen_chunks:
                            chunks.append(chunk)
                            seen_chunks.add(chunk_id)
            
            # If no exact matches, fall back to hybrid search
            if not chunks:
                retrieved_chunks = self.document_processor.hybrid_search(query_info["expanded_query"], top_k=7)
                chunks = [chunk for chunk, score in retrieved_chunks]
                retrieval_scores = [score for chunk, score in retrieved_chunks]
            else:
                # Assign default scores for keyword matches
                retrieval_scores = [0.9] * len(chunks)
        
        # Save retrieved chunks for context
        self.last_retrieved_chunks = chunks
        
        if not chunks:
            logger.warning(f"No relevant chunks found for query: {query}")
            response_text = "I couldn't find any relevant information to answer your query. Could you please rephrase or provide more details?"
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "response": {"summary": response_text, "referenced_images": [], "referenced_tables": []},
                "retrieved_chunks": []
            })
            
            return {
                "success": False,
                "query": query,
                "response": {"summary": response_text, "referenced_images": [], "referenced_tables": []},
                "retrieved_chunks": [],
                "retrieval_scores": []
            }
        
        logger.info(f"Found {len(chunks)} relevant chunks")
        
        # Collect referenced images and tables
        referenced_images = {}
        referenced_tables = {}
        
        for chunk in chunks:
            # Collect images
            for img_id in chunk.images:
                img = self.document_processor.get_image_by_id(img_id)
                if img:
                    referenced_images[img_id] = img
                    
            # Collect tables
            for table_id in chunk.tables:
                table = self.document_processor.get_table_by_id(table_id)
                if table:
                    referenced_tables[table_id] = table
        
        # Generate concise summary
        response = self.summary_generator.generate_summary(
            chunks, 
            query_info,
            list(referenced_images.values()),
            list(referenced_tables.values())
        )
        
        # Extract key terms for additional context
        all_content = " ".join([chunk.text for chunk in chunks])
        key_terms = self.document_processor.extract_key_terms(all_content)
        
        # Structure the final response
        response["key_terms"] = key_terms
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "response": response,
            "retrieved_chunks": chunks
        })
        
        # Limit conversation history size
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return {
            "success": True,
            "query": query,
            "response": response,
            "retrieved_chunks": chunks,
            "retrieval_scores": retrieval_scores,
            "referenced_images": referenced_images,
            "referenced_tables": referenced_tables
        }
    
    def generate_pdf_report(self, query: str, response: Dict[str, Any], 
                           retrieved_chunks: List[DocumentChunk],
                           referenced_images: Dict[str, ImageData] = None,
                           referenced_tables: Dict[str, TableData] = None,
                           output_path: str = None) -> str:
        """Generate a PDF report of the response."""
        logger.info("Generating enhanced PDF report")
        
        try:
            # Get document metadata for sources
            document_metadata = {}
            for chunk in retrieved_chunks:
                if chunk.doc_id not in document_metadata:
                    metadata = self.document_processor.get_document_metadata(chunk.doc_id)
                    if metadata:
                        document_metadata[chunk.doc_id] = metadata
            
            # Generate PDF report
            pdf_path = self.pdf_generator.generate_pdf_report(
                query=query,
                response=response,
                retrieved_chunks=retrieved_chunks,
                images=referenced_images,
                tables=referenced_tables,
                document_metadata=document_metadata,
                output_path=output_path
            )
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self, max_items: int = None) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        if max_items:
            return self.conversation_history[-max_items:]
        return self.conversation_history
    
    def evaluate_response(self, query: str, response: Dict[str, Any], 
                         retrieved_chunks: List[DocumentChunk],
                         reference_summary: str = None) -> Dict[str, Any]:
        """Evaluate a response for the given query."""
        return self.evaluation.evaluate_response(
            query=query,
            response=response,
            retrieved_chunks=retrieved_chunks,
            reference_summary=reference_summary
        )

# Interactive Web Interface
class StreamlitApp:
    """Streamlit-based web interface for the RAG system."""
    
    def __init__(self, rag_system: MultimodalRAGSystem):
        """Initialize the Streamlit app."""
        self.rag_system = rag_system
    
    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(page_title="Multimodal RAG System", layout="wide")
        
        # Title and description
        st.title("Enhanced Multimodal RAG System")
        st.write("Ask questions about documents in the knowledge base.")
        
        # Sidebar
        with st.sidebar:
            st.header("Knowledge Base")
            
            # Knowledge base info
            num_docs = len(self.rag_system.document_processor.documents)
            num_chunks = sum(len(chunks) for chunks in self.rag_system.document_processor.document_chunks.values())
            num_images = len(self.rag_system.document_processor.images)
            num_tables = len(self.rag_system.document_processor.tables)
            
            st.write(f"Documents: {num_docs}")
            st.write(f"Chunks: {num_chunks}")
            st.write(f"Images: {num_images}")
            st.write(f"Tables: {num_tables}")
            
            # Process documents button
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    self.rag_system.document_processor.process_all_documents(incremental=False)
                st.success("Documents processed successfully!")
                
            # Clear history button
            if st.button("Clear Conversation History"):
                self.rag_system.clear_conversation_history()
                st.success("Conversation history cleared!")
                st.experimental_rerun()
        
        # Main content
        # Query input
        query = st.text_input("Enter your query:", key="query_input")
        col1, col2 = st.columns([1, 10])
        with col1:
            consider_history = st.checkbox("Consider conversation history", value=True)
        with col2:
            search_button = st.button("Search")
            
        # Process query
        if search_button and query:
            with st.spinner("Processing query..."):
                result = self.rag_system.process_query(query, consider_history=consider_history)
                
            # Display response
            if result["success"]:
                # Display summary
                st.markdown("### Response")
                st.write(result["response"]["summary"])
                
                # Key terms
                if result["response"].get("key_terms"):
                    with st.expander("Key Terms"):
                        terms = result["response"]["key_terms"]
                        st.write(", ".join(terms))
                
                # Display sources
                with st.expander("Sources"):
                    # Group sources by document
                    doc_sources = {}
                    for chunk in result["retrieved_chunks"]:
                        if chunk.doc_id not in doc_sources:
                            doc_sources[chunk.doc_id] = []
                        doc_sources[chunk.doc_id].append(chunk)
                        
                    for doc_id, chunks in doc_sources.items():
                        metadata = self.rag_system.document_processor.get_document_metadata(doc_id)
                        if metadata and metadata.title:
                            st.markdown(f"**{metadata.title}** ({doc_id})")
                        else:
                            st.markdown(f"**{doc_id}**")
                            
                        pages = sorted(set(chunk.page_num for chunk in chunks if chunk.page_num > 0))
                        if pages:
                            st.write(f"Pages: {', '.join(str(p) for p in pages)}")
                            
                # Display images
                if result["response"].get("referenced_images") and result.get("referenced_images"):
                    st.markdown("### Referenced Images")
                    image_cols = st.columns(min(3, len(result["response"]["referenced_images"])))
                    
                    for i, img_id in enumerate(result["response"]["referenced_images"]):
                        if img_id in result["referenced_images"]:
                            img_data = result["referenced_images"][img_id]
                            col_idx = i % len(image_cols)
                            
                            with image_cols[col_idx]:
                                try:
                                    # Convert image bytes to PIL Image
                                    img = Image.open(BytesIO(img_data.content))
                                    
                                    # Display image
                                    st.image(img, caption=img_data.caption or f"Image from page {img_data.page_num}")
                                    
                                    # Show OCR text if available
                                    if img_data.ocr_text and len(img_data.ocr_text.strip()) > 0:
                                        with st.expander("Text from image"):
                                            st.write(img_data.ocr_text)
                                except Exception as e:
                                    st.error(f"Error displaying image: {str(e)}")
                
                # Display tables
                if result["response"].get("referenced_tables") and result.get("referenced_tables"):
                    st.markdown("### Referenced Tables")
                    
                    for table_id in result["response"]["referenced_tables"]:
                        if table_id in result["referenced_tables"]:
                            table_data = result["referenced_tables"][table_id]
                            
                            st.markdown(f"**Table from page {table_data.page_num}**")
                            st.dataframe(table_data.table_data)
                            
                # Generate PDF button
                if st.button("Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        pdf_path = self.rag_system.generate_pdf_report(
                            query=query,
                            response=result["response"],
                            retrieved_chunks=result["retrieved_chunks"],
                            referenced_images=result.get("referenced_images"),
                            referenced_tables=result.get("referenced_tables")
                        )
                        
                    if pdf_path:
                        st.success(f"PDF report generated successfully!")
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Download PDF Report",
                                data=f,
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf"
                            )
            else:
                st.warning(result["response"]["summary"])
        
        # Display conversation history
        history = self.rag_system.get_conversation_history()
        if history:
            st.markdown("### Conversation History")
            for i, item in enumerate(history):
                with st.container():
                    st.markdown(f"**You**: {item['query']}")
                    st.markdown(f"**Assistant**: {item['response'].get('summary', '')}")
                    st.markdown("---")



def main():
    """Main function to run the enhanced RAG system."""
    KNOWLEDGE_BASE_PATH = "knowledge_base_docs"
    
    print(f"Initializing Enhanced Multimodal RAG system with knowledge base path: {KNOWLEDGE_BASE_PATH}")
    
    # Initialize the RAG system
    rag_system = MultimodalRAGSystem(KNOWLEDGE_BASE_PATH)
    
    # Find and process documents
    new_files = rag_system.document_processor.find_new_files()
    if new_files:
        print(f"Found {len(new_files)} new files. Processing...")
        rag_system.document_processor.process_all_documents()
    else:
        print("No new files to process.")
    
    print("\nSetup completed. You can now interact with the RAG system.")
    print("Type 'exit' to quit, 'clear' to clear conversation history, 'pdf' to generate a PDF of the last response.")
    
    while True:
        query = input("\nEnter your query: ")
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'clear':
            rag_system.clear_conversation_history()
            print("Conversation history cleared.")
            continue
        elif query.lower() == 'pdf' and rag_system.conversation_history:
            last_interaction = rag_system.conversation_history[-1]
            try:
                pdf_path = rag_system.generate_pdf_report(
                    last_interaction["query"],
                    last_interaction["response"],
                    last_interaction["retrieved_chunks"]
                )
                print(f"\nPDF report generated at: {pdf_path}")
            except Exception as e:
                print(f"\nError generating PDF report: {str(e)}")
            continue
        elif query.lower() == 'web':
            # Launch the web interface
            print("Launching web interface...")
            app = StreamlitApp(rag_system)
            import streamlit.web.cli as stcli
            import sys
            sys.argv = ["streamlit", "run", "app.py"]
            sys.exit(stcli.main())
        
        print("\nProcessing your query...\n")
        
        # Process the query
        result = rag_system.process_query(query, verbose=True)
        
        if result["success"]:
            print("=" * 80)
            print("RESPONSE:")
            print("-" * 80)
            print(result["response"]["summary"])
            print("=" * 80)
            
            # Show source documents
            docs = list(set([chunk.doc_id for chunk in result["retrieved_chunks"]]))
            print("\nSources:", ", ".join(docs))
            
            # Show top 3 chunks with scores
            print("\nTop retrieved chunks:")
            for i, (chunk, score) in enumerate(zip(result["retrieved_chunks"][:3], result["retrieval_scores"][:3])):
                print(f"{i+1}. [{chunk.doc_id}] (Score: {score:.2f}): {chunk.text[:100]}...")
            
            # Show referenced images and tables
            if result["response"].get("referenced_images"):
                print(f"\nReferenced Images: {len(result['response']['referenced_images'])}")
            
            if result["response"].get("referenced_tables"):
                print(f"\nReferenced Tables: {len(result['response']['referenced_tables'])}")
        else:
            print(f"Error: {result['response']['summary']}")

# For use in a Jupyter notebook
def setup_rag_system(knowledge_base_path: str = "knowledge_base_docs", auto_process: bool = True):
    """Create and set up a RAG system for use in a notebook."""
    # Initialize the RAG system
    rag_system = MultimodalRAGSystem(knowledge_base_path, auto_process=auto_process)
    
    if auto_process:
        print(f"Initialized RAG system with {len(rag_system.document_processor.documents)} documents")
        
        # Count elements
        num_chunks = sum(len(chunks) for chunks in rag_system.document_processor.document_chunks.values())
        num_images = len(rag_system.document_processor.images)
        num_tables = len(rag_system.document_processor.tables)
        
        print(f"Total chunks: {num_chunks}")
        print(f"Total images: {num_images}")
        print(f"Total tables: {num_tables}")
    else:
        print("RAG system initialized without processing documents")
        print("Call rag_system.document_processor.process_all_documents() to process documents")
    
    return rag_system

def launch_web_interface(rag_system):
    """Launch the web interface for the RAG system."""
    app = StreamlitApp(rag_system)
    app.run()

if __name__ == "__main__":
    # Install required packages if running for the first time
    install_packages()
    
    # Check if this script is being run by Streamlit
    if 'streamlit' in sys.modules:
        # If being run by Streamlit, create rag_system and run the app
        rag_system = MultimodalRAGSystem("knowledge_base_docs", auto_process=True)
        app = StreamlitApp(rag_system)
        app.run()
    else:
        # Otherwise, run the main CLI interface
        main()          