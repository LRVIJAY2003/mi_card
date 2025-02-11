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
import os
from typing import List, Dict, Any
import json
import datetime
from IPython.display import display
import ipywidgets as widgets
from vertexai.language_models import TextGenerationModel
import PyPDF2
import docx
import nltk
from bs4 import BeautifulSoup
import re

class DocumentProcessor:
    """Handles processing of different document formats."""
    
    def __init__(self):
        """Initialize document processor with supported formats."""
        self._supported_formats = {'.txt', '.pdf', '.docx', '.html'}
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document and return structured content."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self._supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        content = self._get_content(file_path)
        sections = self._split_into_sections(content)
        
        return {
            'content': content,
            'sections': sections,
            'keywords': self._extract_keywords(content)
        }
    
    def _get_content(self, file_path: str) -> str:
        """Extract text content based on file type."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._process_pdf(file_path)
        elif file_ext == '.docx':
            return self._process_docx(file_path)
        elif file_ext == '.html':
            return self._process_html(file_path)
        else:  # .txt
            return self._process_txt(file_path)
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return self._clean_text(text)
    
    def _process_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        doc = docx.Document(file_path)
        return self._clean_text("\n".join([paragraph.text for paragraph in doc.paragraphs]))
    
    def _process_html(self, file_path: str) -> str:
        """Extract text from HTML."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return self._clean_text(soup.get_text())
    
    def _process_txt(self, file_path: str) -> str:
        """Read text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return self._clean_text(file.read())
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = text.strip()
        return text
    
    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split text into manageable sections."""
        sentences = nltk.sent_tokenize(text)
        sections = []
        current_section = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > 1000:  # Max section size
                if current_section:
                    sections.append({
                        'content': ' '.join(current_section)
                    })
                current_section = [sentence]
                current_length = sentence_length
            else:
                current_section.append(sentence)
                current_length += sentence_length
        
        if current_section:
            sections.append({
                'content': ' '.join(current_section)
            })
        
        return sections
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text."""
        words = text.lower().split()
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return [word for word in words if word not in stop_words and len(word) > 3]

class KnowledgeBaseSystem:
    """Manages the knowledge base and handles queries."""
    
    def __init__(self, project_id: str = 'cloud-workspace-poc-51731', location: str = 'us-central1'):
        """Initialize the knowledge base system."""
        self.project_id = project_id
        self.location = location
        self.model = TextGenerationModel('gemini-1.0-pro-001')
        self.knowledge_base = {}
        self.knowledge_base_path = 'knowledge_base.json'
        self.doc_processor = DocumentProcessor()
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load existing knowledge base from file."""
        try:
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print(f"Loaded {len(self.knowledge_base)} documents")
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            self.knowledge_base = {}
    
    def save_knowledge_base(self):
        """Save knowledge base to file."""
        try:
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
    
    def add_document_to_knowledge_base(self, file_path: str):
        """Process and add document to knowledge base."""
        try:
            doc_id = os.path.basename(file_path)
            doc_info = self.doc_processor.process_document(file_path)
            
            self.knowledge_base[doc_id] = {
                'doc_id': doc_id,
                'content': doc_info['content'],
                'sections': doc_info['sections'],
                'keywords': doc_info['keywords'],
                'added_date': datetime.datetime.now().isoformat()
            }
            
            self.save_knowledge_base()
            print(f"Successfully added {doc_id}")
            
        except Exception as e:
            print(f"Error adding document {file_path}: {str(e)}")
    
    def search_knowledge_base(self, query: str, max_relevant_chunks: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant sections."""
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
            print(f"Error searching knowledge base: {str(e)}")
            return []
    
    def generate_response(self, query: str, additional_context: str = "") -> str:
        """Generate response using Gemini model."""
        try:
            relevant_docs = self.search_knowledge_base(query)
            
            if not relevant_docs:
                return "No relevant information found in the knowledge base."
            
            context = "\n\n".join([f"From document ({doc['doc_id']}): {doc['content']}"
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
   - Do not make assumptions or add information not present in the source

2. Response Structure:
   - Start with a direct answer to the query
   - Provide relevant details and context
   - Include step by step instructions if applicable
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
            return f"Error generating response: {str(e)}"

class KnowledgeBaseInterface:
    """Provides user interface for interacting with knowledge base."""
    
    def __init__(self, kb_system: KnowledgeBaseSystem):
        """Initialize the interface."""
        self.kb_system = kb_system
        self.setup_interface()
    
    def setup_interface(self):
        """Set up the UI components."""
        # File upload widget
        self.file_upload = widgets.FileUpload(
            accept='.pdf,.txt,.docx,.html',
            multiple=True,
            description='Upload Documents'
        )
        
        # Query input
        self.query_input = widgets.Text(
            value='',
            placeholder='Enter your query here...',
            description='Query:',
            layout={'width': '800px', 'height': '100px'}
        )
        
        # Output area
        self.output_area = widgets.Output()
        
        # Search button
        self.search_button = widgets.Button(description='Search')
        
        # Set up event handlers
        self.file_upload.observe(self.handle_upload, names='value')
        self.search_button.on_click(self.handle_search)
        
        # Display UI components
        display(widgets.VBox([
            widgets.HTML("<h2>Knowledge Base Query System</h2>"),
            self.file_upload,
            self.query_input,
            self.search_button,
            self.output_area
        ]))
    
    def handle_upload(self, change):
        """Handle file upload events."""
        with self.output_area:
            self.output_area.clear_output()
            try:
                for filename, file_info in change['new'].items():
                    content = file_info['content']
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{filename}"
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                    
                    # Process and add to knowledge base
                    self.kb_system.add_document_to_knowledge_base(temp_path)
                    
                    # Clean up temporary file
                    os.remove(temp_path)
                    
                print("Documents uploaded successfully!")
                    
            except Exception as e:
                print(f"Error uploading documents: {str(e)}")
    
    def handle_search(self, button):
        """Handle search button clicks."""
        with self.output_area:
            self.output_area.clear_output()
            query = self.query_input.value.strip()
            
            if query:
                try:
                    response = self.kb_system.generate_response(query)
                    print("Query:", query)
                    print("\nResponse:")
                    print(response)
                except Exception as e:
                    print(f"Error searching: {str(e)}")
            else:
                print("Please enter a query first.")

# Usage
if __name__ == "__main__":
    kb_system = KnowledgeBaseSystem()
    interface = KnowledgeBaseInterface(kb_system)