## 📚 SmartPDF Chat – RAG Chatbot con PDFs

SmartPDF Chat es una aplicación de Retrieval-Augmented Generation (RAG) que permite subir documentos PDF y hacer preguntas sobre su contenido utilizando modelos de lenguaje y búsqueda semántica.

La aplicación está construida con Streamlit, LangChain, ChromaDB y Groq, permitiendo analizar documentos y generar respuestas basadas en su contenido.

___

Características

* 📄 Subida de múltiples documentos PDF

* 🔎 Búsqueda semántica con embeddings

* 🧠 Respuestas generadas con LLM (Groq / Llama)

* 💬 Chat conversacional con memoria

* 📚 Visualización de archivos fuente utilizados

* ⚙️ Configuración dinámica de parámetros:

  * Número de documentos recuperados (k)

  * Tipo de búsqueda (similarity / mmr)

  * Temperatura del modelo

* 🔄 Botón para reiniciar la base vectorial

* ⚡ Procesamiento eficiente de documentos
___

### 🧠 Arquitectura RAG

El flujo del sistema es el siguiente:

```
PDF Upload
     │
     ▼
Document Loader (PyPDFLoader)
     │
     ▼
Text Splitter
     │
     ▼
Embeddings (HuggingFace)
     │
     ▼
Vector Database (ChromaDB)
     │
     ▼
Retriever
     │
     ▼
LLM (Groq - Llama 3)
     │
     ▼
Answer + Source Documents 
```
---

### 🛠️ Tecnologías utilizadas
* Python

* Streamlit

* LangChain

* ChromaDB

* HuggingFace Embeddings

* Groq LLM (Llama 3.3 70B)

* PyPDFLoader

___

📂 Estructura del proyecto
```
.
├── app_mod.py
├── rag_utility_mod.py
├── .env
├── requirements.txt
└── README.md
```
app_mod.py

Interfaz principal de la aplicación:

* Upload de PDFs

* Interfaz de chat

* Control de parámetros

* Visualización de historial

rag_utility_mod.py

Módulo que implementa la lógica RAG:

* Procesamiento de documentos

* Generación de embeddings

* Vector database (Chroma)

* Retriever

* Cadena conversacional con LangChain
---

### 📦 Instalación

Clona el repositorio:
``` bash
git clone https://github.com/tu_usuario/smartpdf-chat.git
cd smartpdf-chat
```
nstala las dependencias:
``` bash
pip install -r requirements.txt
```
---

### 🔑 Variables de entorno

Crea un archivo .env en la raíz del proyecto:
``` bash
GROQ_API_KEY=tu_api_key
```
---

### ▶️ Ejecutar la aplicación

Inicia el servidor de Streamlit:
``` bash
streamlit run app_mod.py
```
---

### 📄 Uso de la aplicación

1. Subir hasta 3 documentos PDF

2. Procesarlos en la base vectorial

3. Hacer preguntas sobre el contenido

4. El sistema recupera los fragmentos relevantes

5. El modelo genera una respuesta basada en esos documentos

6. También se muestran los archivos fuente utilizados para generar la respuesta.
---

### ⚙️ Parámetros configurables
**Número de documentos recuperados (k)**

* Controla cuántos fragmentos se usan como contexto.

* Mayor valor → más contexto

Menor valor → respuestas más precisas

**Tipo de búsqueda**

similarity

* Basado en similitud de embeddings

mmr (Maximal Marginal Relevance)

* Balancea relevancia y diversidad

**Temperatura**

Controla la creatividad del modelo.
``` bash
0.0 → Respuestas más determinísticas
1.0 → Respuestas más creativas
```
---

### 🧹 Reinicio del sistema

El botón Reset all data:

* Borra la base vectorial

* Limpia el historial de chat

* Permite cargar nuevos documentos
---

### 📜 Licencia

MIT License

---

💡 Este proyecto demuestra cómo construir un sistema RAG completo para consultar documentos con IA utilizando herramientas modernas del ecosistema LangChain + LLMs.

