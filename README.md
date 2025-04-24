# Future Minds Chatbot GUI 🤖

This is a Streamlit-based user interface for the multi-agent GenAI chatbot developed for the CodeJam Future Minds competition.

The chatbot uses:
- 🧠 **Gemini 1.5 Flash** for answer generation
- 🔎 **FAISS** for vector-based document search
- 📖 Grade 11 History textbook as its source

## 🚀 Features

- Ask history questions in plain English
- Get AI-generated answers with cited pages
- See the exact context chunks used
- Fully local and fast to run

## 📁 Folder Structure

```
📦 future_minds_chatbot_gui/
├── chatbot_app.py         # Streamlit app code
├── requirements.txt       # All Python dependencies
└── data/                  # Place your textbook PDF here
```

## ⚙️ Setup Instructions

1. Install dependencies (preferably inside a virtual environment):
```bash
pip install -r requirements.txt
```

2. Add your textbook and queries:
- Place `grade-11-history-text-book.pdf` inside the `data/` folder

3. Launch the app:
```bash
streamlit run chatbot_app.py
```

4. Visit the app in your browser at `http://localhost:8501`

## 🏁 Built For

✨ CodeJam Future Minds GenAI App Challenge (April 2025) ✨
