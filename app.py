
import os
import json
import streamlit as st
import torch
import json_repair

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

try:
    from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 AI Career Coach")
st.write(
    "ارفع الـCV وأجب عن 3 أسئلة، وسيقوم الذكاء الاصطناعي بتحليل مستواك "
    "والمهارات الناقصة واقتراح Roadmap ووظائف مناسبة — لأي مجال."
)

MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"


# =========================
# Output Schema
# =========================
response_schemas = [
    ResponseSchema(
        name="Current Skills",
        description="قائمة مفصلة بالمهارات التقنية، الأدوات، الشهادات والخبرات المستخرجة من CV وإجابات المستخدم.",
    ),
    ResponseSchema(
        name="Missing Skills",
        description="قائمة بالمهارات والأدوات الناقصة والمطلوبة لسوق العمل في المجال أو الدور المستهدف.",
    ),
    ResponseSchema(
        name="Recommended Roadmap",
        description="خارطة طريق تعليمية مقسمة إلى مراحل عملية واضحة ومناسبة لمستوى المستخدم.",
    ),
    ResponseSchema(
        name="Suggested Jobs",
        description="الوظائف المناسبة لمستوى المستخدم الحالي مع ذكر سبب التوافق.",
    ),
    ResponseSchema(
        name="Improvement Plan",
        description="خطة عملية لتطوير المهارات وبناء المشاريع والبورتفوليو والاستعداد لسوق العمل.",
    ),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


# =========================
# Model Loading
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    else:
        # CPU fallback. Mistral-Nemo is large, so GPU is strongly recommended.
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    return tokenizer, model


def generate_response(prompt_text, tokenizer, model, max_new_tokens=1200):
    formatted_prompt = f"<s>[INST] {prompt_text} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# =========================
# Helpers
# =========================
def extract_cv_text(uploaded_file):
    temp_path = "temp_cv.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    try:
        os.remove(temp_path)
    except OSError:
        pass

    return "\n".join(page.page_content for page in pages), len(pages)


def build_rag(cv_text, target_role, skills_tools, projects_experience):
    user_documents = [
        Document(
            page_content=f"User Full CV Details:\n{cv_text[:4000]}",
            metadata={"category": "cv_data"},
        ),
        Document(
            page_content=(
                f"User Evaluation Q&A:\n"
                f"Target Role / Field: {target_role}\n"
                f"Current Skills, Tools & Technologies: {skills_tools}\n"
                f"Projects, Certifications & Experience: {projects_experience}"
            ),
            metadata={"category": "evaluation_qa"},
        ),
    ]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(user_documents, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 2})


def create_prompt(rag_context, target_role):
    coach_prompt = PromptTemplate(
        template="""
أنت مستشار توظيف وخبير Career Coach محترف ويمكنك التعامل مع أي مجال مهني،
وليس مجالاً تقنياً أو Backend فقط.

بناءً على ملف المستخدم وإجاباته المسترجعة من الـRAG:
{rag_context}

المجال أو الوظيفة المستهدفة:
{target_role}

المطلوب:
1. استخرج المهارات والخبرات والشهادات والأدوات الفعلية الموجودة في البيانات.
2. حدد المهارات الناقصة مقارنة بمتطلبات سوق العمل للمجال/الوظيفة المستهدفة.
3. ضع خارطة طريق عملية ومقسمة لمراحل تناسب مستوى المستخدم.
4. اقترح وظائف مناسبة للمستوى الحالي مع سبب التوافق.
5. ضع خطة تحسين عملية لبناء المهارات والمشاريع والبورتفوليو.

قواعد مهمة:
- لا تفترض أن المجال هو Backend أو Cybersecurity أو أي مجال محدد.
- استخدم المجال الذي أدخله المستخدم كما هو.
- لا تخترع شهادات أو خبرات أو مهارات غير موجودة في بيانات المستخدم.
- إذا كانت معلومة غير موجودة، تعامل معها على أنها غير معروفة.
- الإجابة يجب أن تكون JSON صالح فقط يبدأ بـ {{ وينتهي بـ }}.
- املأ الحقول بالتفصيل في شكل نقاط واضحة ومباشرة.

{format_instructions}
""",
        input_variables=["rag_context", "target_role"],
        partial_variables={"format_instructions": format_instructions},
    )

    return coach_prompt.format(
        rag_context=rag_context,
        target_role=target_role,
    )


def parse_output(raw_output):
    try:
        return output_parser.parse(raw_output)
    except Exception:
        try:
            parsed = json_repair.loads(raw_output)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            return parsed
        except Exception:
            return {"Improvement Plan": [raw_output]}


def display_report(data):
    sections = [
        ("Current Skills", "🛠️ المهارات الحالية"),
        ("Missing Skills", "🎯 المهارات الناقصة"),
        ("Recommended Roadmap", "🗺️ خارطة الطريق"),
        ("Suggested Jobs", "💼 الوظائف المقترحة"),
        ("Improvement Plan", "🚀 خطة التحسين"),
    ]

    for key, title in sections:
        if key not in data:
            continue

        st.subheader(title)
        value = data[key]

        if isinstance(value, list):
            for item in value:
                st.markdown(f"- {str(item).strip().strip(chr(34) + chr(39))}")
        elif isinstance(value, dict):
            st.json(value)
        else:
            st.markdown(str(value))


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("AI Career Coach")
    st.write("Model:")
    st.code(MODEL_NAME)
    st.write(
        "يفضل تشغيل التطبيق على جهاز/سيرفر يحتوي على GPU لأن Mistral-Nemo "
        "موديل كبير."
    )


# =========================
# Main Inputs
# =========================
uploaded_file = st.file_uploader(
    "📄 ارفع الـCV بصيغة PDF",
    type=["pdf"],
)

st.markdown("### 📝 التقييم المهني")

target_role = st.text_input(
    "1️⃣ المجال أو الوظيفة المستهدفة",
    placeholder="مثال: Cybersecurity, Data Analyst, Frontend Developer, Accountant...",
)

skills_tools = st.text_area(
    "2️⃣ ما المهارات والأدوات والتقنيات التي تستخدمها حالياً؟",
    placeholder="اكتب المهارات، الأدوات، البرامج أو التقنيات التي تعرف استخدامها...",
    height=120,
)

projects_experience = st.text_area(
    "3️⃣ ما المشاريع أو الشهادات أو اللابات أو الخبرات التي لديك؟",
    placeholder="اذكر المشاريع، الشهادات، التدريب، اللابات، الخبرات العملية أو الأكاديمية...",
    height=150,
)

analyze = st.button("🚀 Analyze My Career", type="primary", use_container_width=True)


# =========================
# Analysis
# =========================
if analyze:
    if uploaded_file is None:
        st.error("❌ من فضلك ارفع ملف الـCV أولاً.")
        st.stop()

    if not target_role.strip():
        st.error("❌ من فضلك اكتب المجال أو الوظيفة المستهدفة.")
        st.stop()

    if not skills_tools.strip():
        st.error("❌ من فضلك أجب عن السؤال الثاني.")
        st.stop()

    if not projects_experience.strip():
        st.error("❌ من فضلك أجب عن السؤال الثالث.")
        st.stop()

    try:
        with st.status("🔄 جاري تحليل البيانات...", expanded=True) as status:
            st.write("📄 استخراج النص من الـCV...")
            cv_text, pages_count = extract_cv_text(uploaded_file)
            st.write(f"✅ تم استخراج {pages_count} صفحة.")

            st.write("🧠 تحميل نموذج Mistral-Nemo...")
            tokenizer, model = load_model()
            st.write("✅ تم تجهيز النموذج.")

            st.write("🔎 بناء قاعدة RAG باستخدام FAISS...")
            retriever = build_rag(
                cv_text,
                target_role,
                skills_tools,
                projects_experience,
            )

            st.write("🔍 استرجاع المعلومات المهمة...")
            retrieved_docs = retriever.invoke(
                f"Extract skills, experience, certifications and tools relevant to {target_role}"
            )
            rag_context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            st.write("🤖 إنشاء التقرير بواسطة Mistral...")
            full_prompt = create_prompt(rag_context, target_role)
            raw_output = generate_response(
                full_prompt,
                tokenizer,
                model,
            )

            st.write("🧹 معالجة وتحويل النتيجة...")
            parsed_data = parse_output(raw_output)

            status.update(
                label="✅ تم الانتهاء من التحليل!",
                state="complete",
                expanded=False,
            )

        st.divider()
        st.header("📋 AI Career Coach Report")
        display_report(parsed_data)

        with st.expander("🔍 View Raw Model Output"):
            st.code(raw_output, language="json")

    except Exception as e:
        st.error("❌ حدث خطأ أثناء تشغيل التحليل.")
        st.exception(e)
