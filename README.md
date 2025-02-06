![App Brewery Banner](https://github.com/londonappbrewery/Images/blob/master/AppBreweryBanner.png)

# Mi Card

## Our Goal

Now that you've seen how to create a Flutter app entirely from scratch, we're going to go further and learn more about how to design user interfaces for Flutter apps.

## What you will create

Mi Card is a personal business card. Imagine every time you wanted to give someone your contact details or your business card but you didn't have it on you. Well, now you can get them to download your business card as an app.

## What you will learn

* How to create Stateless Widgets
* What is the difference between hot reload and hot refresh and running an app from cold
* How to use Containers to lay out your UI
* How to use Columns and Rows to position your UI elements
* How to add custom fonts
* How to add Material icons
* How to style Text widgets
* How to read and use Flutter Documentation



>This is a companion project to The App Brewery's Complete Flutter Development Bootcamp, check out the full course at [www.appbrewery.co](https://www.appbrewery.co/)

![End Banner](https://github.com/londonappbrewery/Images/blob/master/readme-end-banner.png)


![End Banner](./mi_card.png)


# Install required packages
!pip install -q google-cloud-aiplatform PyPDF2 python-docx nltk beautifulsoup4 markdown

import os
from google.cloud import aiplatform
import PyPDF2
import json
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from docx import Document
import markdown
import ipywidgets as widgets
from IPython.display import display, HTML
import logging

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class DocumentProcessor:
    """Handles document processing and text extraction"""
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.txt': self._process_text,
            '.docx': self._process_docx,
            '.md': self._process_markdown,
            '.html': self._process_html
        }

    def process_document(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        content = self.supported_formats[file_ext](file_path)
        sections = self._split_into_sections(content)
        return {
            'content': content,
            'sections': sections,
            'keywords': self._extract_keywords(content)
        }

    def _process_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return self._clean_text(text)

    def _process_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return self._clean_text(file.read())

    def _process_docx(self, file_path):
        doc = Document(file_path)
        return self._clean_text('\n'.join([paragraph.text for paragraph in doc.paragraphs]))

    def _process_markdown(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            html_content = markdown.markdown(md_content)
            return self._clean_text(BeautifulSoup(html_content, 'html.parser').get_text())

    def _process_html(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return self._clean_text(soup.get_text())

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def _split_into_sections(self, text):
        sentences = sent_tokenize(text)
        sections = []
        current_section = []
        
        for sentence in sentences:
            current_section.append(sentence)
            if len(' '.join(current_section)) >= 1000:
                sections.append({
                    'content': ' '.join(current_section)
                })
                current_section = []
        
        if current_section:
            sections.append({
                'content': ' '.join(current_section)
            })
        
        return sections

    def _extract_keywords(self, text):
        words = text.lower().split()
        stop_words = set(stopwords.words('english'))
        return list(set([word for word in words if word not in stop_words and len(word) > 3]))

class KnowledgeBaseSystem:
    """Manages the knowledge base and handles query processing"""
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Get the latest model
        self.model = aiplatform.GenerativeModel.from_pretrained("gemini-pro")
        
        self.knowledge_base = {}
        self.knowledge_base_path = 'knowledge_base.json'
        self.doc_processor = DocumentProcessor()
        self.load_knowledge_base()

    def load_knowledge_base(self):
        try:
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print(f"Loaded {len(self.knowledge_base)} documents")
        except Exception as e:
            logging.error(f"Error loading knowledge base: {str(e)}")
            self.knowledge_base = {}

    def save_knowledge_base(self):
        try:
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving knowledge base: {str(e)}")

    def add_document_to_knowledge_base(self, file_path):
        try:
            doc_id = os.path.basename(file_path)
            doc_info = self.doc_processor.process_document(file_path)
            
            self.knowledge_base[doc_id] = {
                'content': doc_info['content'],
                'sections': doc_info['sections'],
                'keywords': doc_info['keywords'],
                'added_date': datetime.now().isoformat()
            }
            
            self.save_knowledge_base()
            print(f"Successfully added {doc_id}")
            
        except Exception as e:
            logging.error(f"Error adding document {file_path}: {str(e)}")

    def search_knowledge_base(self, query, max_relevant_chunks=3):
        try:
            query_keywords = set(self.doc_processor._extract_keywords(query))
            relevant_sections = []
            
            for doc_id, doc_info in self.knowledge_base.items():
                for section in doc_info['sections']:
                    section_keywords = set(self.doc_processor._extract_keywords(section['content']))
                    relevance_score = len(query_keywords.intersection(section_keywords))
                    
                    if relevance_score > 0:
                        relevant_sections.append({
                            'doc_id': doc_id,
                            'content': section['content'],
                            'score': relevance_score
                        })
            
            relevant_sections.sort(key=lambda x: x['score'], reverse=True)
            return relevant_sections[:max_relevant_chunks]
            
        except Exception as e:
            logging.error(f"Error searching knowledge base: {str(e)}")
            return []

    def generate_response(self, query, additional_context=""):
        try:
            relevant_docs = self.search_knowledge_base(query)
            
            if not relevant_docs:
                return "No relevant information found in the knowledge base."

            context = "\n\n".join([f"From document {doc['doc_id']}:\n{doc['content']}" 
                                 for doc in relevant_docs])

            prompt = f"""
            User Query: {query}

            Additional Context (if provided): {additional_context}

            Relevant Information from Knowledge Base:
            {context}

            Please provide a comprehensive response following these guidelines:

            1. Answer Accuracy:
               - Use ONLY information from the provided documentation
               - If information is insufficient, clearly state what cannot be answered
               - Do not make assumptions or add information not present in the sources

            2. Response Structure:
               - Start with a direct answer to the query
               - Provide relevant details and context
               - Include step-by-step instructions if applicable
               - List any prerequisites or dependencies mentioned

            3. Source Attribution:
               - Reference specific documents for key information
               - Indicate which document contains which part of the answer

            4. Technical Details:
               - Include any specific technical parameters mentioned
               - Note any version requirements or compatibility issues
               - Highlight any warnings or important considerations

            5. Additional Considerations:
               - Mention any related topics that might be relevant
               - Note any limitations or edge cases
               - Suggest related queries if applicable

            Remember: Base your response ONLY on the provided documentation. If certain aspects of the query cannot be answered from the available information, explicitly state this limitation.
            """

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

class KnowledgeBaseInterface:
    """Provides the user interface for the knowledge base system"""
    def __init__(self, kb_system):
        self.kb_system = kb_system
        self.setup_interface()

    def setup_interface(self):
        self.query_input = widgets.Textarea(
            placeholder='Enter your question here...',
            layout={'width': '800px', 'height': '100px'}
        )
        
        self.file_upload = widgets.FileUpload(
            accept='.pdf,.txt,.docx,.md,.html',  # Updated to include MD and HTML
            multiple=True,
            description='Upload Documents'
        )
        
        self.search_button = widgets.Button(description='Search')
        self.output_area = widgets.Output()
        
        self.file_upload.observe(self.handle_upload, names='value')
        self.search_button.on_click(self.handle_search)
        
        display(HTML("<h2>Knowledge Base Query System</h2>"))
        display(self.query_input)
        display(widgets.HBox([self.search_button, self.file_upload]))
        display(self.output_area)

    def handle_upload(self, change):
        with self.output_area:
            self.output_area.clear_output()
            try:
                for filename, data in change['new'].items():
                    with open(filename, 'wb') as f:
                        f.write(data['content'])
                    self.kb_system.add_document_to_knowledge_base(filename)
                    os.remove(filename)
                print("Documents uploaded successfully!")
            except Exception as e:
                print(f"Error uploading documents: {str(e)}")

    def handle_search(self, button):
        with self.output_area:
            self.output_area.clear_output()
            query = self.query_input.value.strip()
            if query:
                print("Searching...")
                response = self.kb_system.generate_response(query)
                print("\nQuery:", query)
                print("\nResponse:")
                print(response)
            else:
                print("Please enter a query first.")
