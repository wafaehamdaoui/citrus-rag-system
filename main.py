import os
import sys
from pathlib import Path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np

# RAG Integration Frameworks
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

app = Flask(__name__)

# --- CONFIGURATION FROM BOTH SCRIPTS ---
PERSIST_PATH = "./qdrant_citrus_db"
COLLECTION_NAME = "citrus-treatments"
DOCS_DIR = "./citrus_docs"
IMAGE_SIZE = 256
class_names = ['blackspot', 'canker', 'grenning', 'healthy']

# Ensure runtime upload & documentation directories exist
os.makedirs('static', exist_ok=True)
Path(DOCS_DIR).mkdir(exist_ok=True)


# --- DEEP LEARNING MODEL INITIALIZATION ---
@tf.keras.utils.register_keras_serializable()
class SymmetryAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SymmetryAttention, self).__init__(**kwargs)
        self.alpha = None
        self.debug_outputs = {}

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(1,), initializer='ones', trainable=True)

    def call(self, feature_map):
        shape = tf.shape(feature_map)
        mid = shape[2] // 2
        F_left = feature_map[:, :, :mid, :]
        F_right = feature_map[:, :, -mid:, :]
        F_right_flipped = tf.reverse(F_right, axis=[2])

        min_width = tf.minimum(tf.shape(F_left)[2], tf.shape(F_right_flipped)[2])
        F_left = F_left[:, :, :min_width, :]
        F_right_flipped = F_right_flipped[:, :, :min_width, :]

        A = tf.abs(F_left - F_right_flipped)
        A_mirror = tf.reverse(A, axis=[2])
        A_concat = tf.concat([A, A_mirror], axis=2)

        M = tf.reduce_mean(A_concat, axis=-1, keepdims=True)
        M = tf.keras.activations.sigmoid(M)

        return feature_map * (1 + self.alpha * M)

    def get_config(self):
        config = super(SymmetryAttention, self).get_config()
        return config

print("Loading Keras deep learning model...")
model = tf.keras.models.load_model('Proposed_CNN_Updated.keras')


# --- INITIALIZE CORE RAG COMPONENTS ---
print("Initializing local HuggingFace embedding engine & Ollama LLM...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3", temperature=0)

client = QdrantClient(path=PERSIST_PATH)

try:
    # Check if the database already exists
    client.get_collection(collection_name=COLLECTION_NAME)
    vectorstore = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        client=client,
    )
    print("Loaded existing vector database successfully.")
except Exception:
    print(f"Database not found. Scanning '{DOCS_DIR}' for PDF articles...")

    # 1. Load PDFs from the directory
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    
    if not docs:
        print(f"\n [CRITICAL ERROR]: No PDFs found in '{DOCS_DIR}'. Place files there and restart.")
        sys.exit(1)

    # 2. Split text into manageable chunks
    script_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = script_splitter.split_documents(docs)
    print(f"Processed {len(docs)} PDF pages into {len(all_chunks)} text chunks.")

    # 3. Store into Qdrant vector database
    vectorstore = QdrantVectorStore.from_documents(
        all_chunks,
        embedding=embeddings,
        path=PERSIST_PATH,
        collection_name=COLLECTION_NAME,
    )
    print("Collection cleanly built and saved to disk!")

# 4. Construct the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# --- 5. DEFINE PROMPTS CLEARLY ---
# agronomy_template = """You are an expert AI Agronomist specializing in Citrus Pathology. 
# Analyze the question based ONLY on the provided research papers, extension reports, and official documentation guidelines.

# When recommending treatments (chemical, biological, or cultural control):
# 1. Be highly precise regarding active ingredients, dosages, application timing, or sanitary measures if mentioned.
# 2. Clearly state the specific citrus disease targeted.
# 3. If the provided documents offer conflicting strategies or mention regional restrictions, note that distinction.
# 4. If the context does not contain clear, verified treatment protocols for the query, state that verified protocols are unavailable.

# Context Excerpts:
# {context}

# Agronomy Query: 
# {question}

# Expert English Recommendation:"""

# translation_template = """You are a professional, expert translator specializing in agricultural science and plant pathology.
# Translate the following English technical recommendation into flawless, natural, and grammatically correct {target_language}.

# CRITICAL REQUIREMENTS:
# - Your entire final response must be in {target_language}. Do not leak a single English word into the text.
# - Maintain proper Right-to-Left (RTL) formatting if translating to Arabic.
# - Keep standard botanical scientific names in parentheses if necessary, but translate all prose, verbs, and nouns completely.
# - Do not add any introductory pleasantries or meta-commentary.

# Text to Translate:
# {english_recommendation}

# Final Translation:"""

# agronomy_prompt = ChatPromptTemplate.from_template(agronomy_template)
# translation_prompt = ChatPromptTemplate.from_template(translation_template)


# # --- 6. BUILD THE PIPELINE SEQUENTIALLY ---

# from langchain_core.runnables import RunnablePassthrough

# rag_chain = (
#     # A. First, fetch context and keep the user's question and language intact
#     {
#         "context": lambda x: retriever.invoke(x["question"]),
#         "question": lambda x: x["question"],
#         "target_language": lambda x: x["target_language"]
#     }
#     # B. Next, feed that complete dictionary into the English generation step
#     | RunnablePassthrough.assign(
#         english_recommendation=agronomy_prompt | llm | StrOutputParser()
#     )
#     # C. Now the dictionary has 'english_recommendation' and 'target_language' ready for the translation prompt
#     | translation_prompt
#     | llm
#     | StrOutputParser()
# )

# 1. Update LLM component
llm = ChatOllama(model="aya", temperature=0)

# 2. You can use a clean, single-stage prompt template again!
template = """You are an expert AI Agronomist specializing in Citrus Pathology. 
Analyze the question based ONLY on the provided research papers, extension reports, and official documentation guidelines.

CRITICAL REQUIREMENT: You MUST generate your entire final Expert Recommendation response in {target_language}. Do not answer in English.

Context Excerpts:
{context}

Agronomy Query: 
{question}

Expert Recommendation (In {target_language}):"""

prompt = ChatPromptTemplate.from_template(template)

# 3. Bring back the clean, fast single-step chain
rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]), 
        "question": lambda x: x["question"],
        "target_language": lambda x: x["target_language"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- CORE PIPELINE UTILITIES ---
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


# --- LOCALIZED UI MESSAGES ---
UI_MESSAGES = {
    "English": {
        "upload": "Upload an image",
        "no_file_part": "No file part in the request",
        "no_file_selected": "No selected file"
    },
    "French": {
        "upload": "Téléchargez une image",
        "no_file_part": "Pas de fichier dans la requête",
        "no_file_selected": "Aucun fichier sélectionné"
    },
    "Spanish": {
        "upload": "Subir una imagen",
        "no_file_part": "No hay parte de archivo en la solicitud",
        "no_file_selected": "No se ha seleccionado ningún archivo"
    },
    "Arabic": {
        "upload": "يرجى تحميل صورة",
        "no_file_part": "لم يتم إرسال أي ملف في الطلب",
        "no_file_selected": "لم يتم اختيار أي ملف"
    }
}

# --- ENDPOINTS ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Now we safely look for the language sent by the form submission
        target_language = request.form.get('language', 'English')
        lang_content = UI_MESSAGES.get(target_language, UI_MESSAGES["English"])

        if 'file' not in request.files:
            return render_template('index1.html', message=lang_content["no_file_part"], selected_language=target_language)

        file = request.files['file']

        if file.filename == '':
            return render_template('index1.html', message=lang_content["no_file_selected"], selected_language=target_language)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # RUN PREDICTION ONLY (Fast Keras Inference)
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            predicted_class, confidence = predict(img)

            return render_template(
                'index1.html', 
                image_path=filepath, 
                actual_label=os.path.splitext(filename)[0].split('_')[0], 
                predicted_label=predicted_class, 
                confidence=confidence,
                selected_language=target_language # Preserves state after inference
            )

    # FOR GET REQUESTS: Just serve the page. JavaScript will load the language from localStorage.
    return render_template('index1.html', message=None)

@app.route('/get_treatment', methods=['POST'])
def get_treatment():
    data = request.get_json() or {}
    predicted_class = data.get('predicted_label', 'healthy').strip().lower()
    target_language = data.get('language', 'English') # Captures dropdown state

    # Run the RAG system on-demand
    if predicted_class != 'healthy':
        rag_query = f"What is the recommended management protocol and treatment for {predicted_class} in citrus trees?"
        
        # We pass both query context and target localization keyword down the chain
        treatment_recommendation = rag_chain.invoke({
            "question": rag_query,
            "target_language": target_language
        })
    else:
        # Pre-translated status placeholders for healthy leaves
        healthy_responses = {
            "English": "No pathogen control measures required. The sample is evaluated as healthy. Continue normal preventive orchard surveillance.",
            "French": "Aucune mesure de contrôle des agents pathogènes requise. L'échantillon est évalué comme sain. Poursuivre la surveillance normale du verger.",
            "Spanish": "No se requieren medidas de control de patógenos. La muestra se evalúa como sana. Continuar con la vigilancia preventiva normal del huerto.",
            "Arabic": "لا توجد تدابير لمكافحة الآفات مطلوبة. تم تقييم العينة على أنها سليمة تماماً. يرجى الاستمرار في مراقبة البستان."
        }
        treatment_recommendation = healthy_responses.get(target_language, healthy_responses["English"])

    # Return output string back to JavaScript engine
    return treatment_recommendation


if __name__ == '__main__':
    app.run(debug=False)