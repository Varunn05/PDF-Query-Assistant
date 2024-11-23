import os
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pypdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store extracted PDF text
pdf_text = ""

class ChatRequest(BaseModel):
    message: str

def extract_text_from_pdf(file):
    """
    Extract text from uploaded PDF file
    """
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and extract its text
    """
    global pdf_text
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save the uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())
    
    # Extract text from PDF
    try:
        with open(file.filename, "rb") as pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file)
        
        # Remove temporary file
        os.remove(file.filename)
        
        return {
            "message": "PDF uploaded successfully", 
            "text_length": len(pdf_text)
        }
    except Exception as e:
        # Remove temporary file in case of error
        if os.path.exists(file.filename):
            os.remove(file.filename)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-with-pdf/")
async def chat_with_pdf(request: ChatRequest):
    """
    Chat with PDF contents using Google Gemini
    """
    global pdf_text
    
    # Check if PDF is uploaded
    if not pdf_text:
        raise HTTPException(status_code=400, detail="Please upload a PDF first")
    
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare the prompt with PDF context
        full_prompt = f"""
        Context from PDF document:
        {pdf_text[:4000]}

        User Question: {request.message}

        Please provide a helpful and precise answer based on the context above.
        """
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        # Return the assistant's response
        return {
            "response": response.text
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)