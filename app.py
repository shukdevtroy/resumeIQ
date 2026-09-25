"""
Resume / CV Analyzer
---------------------
A Gradio app that compares an uploaded resume (PDF) against a job description
using an LLM served via OpenRouter (default model: inclusionai/ling-3.0-flash-sante:free).

Run with:
    pip install gradio openai pypdf
    python resume_analyzer_app.py
"""

import traceback

import gradio as gr
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MODEL_NAME = "inclusionai/ling-3.0-flash-sante:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are an expert Resume/CV Analyzer and Career Coach with deep experience in
technical recruiting, ATS (Applicant Tracking System) optimization, and career
development across multiple industries.

You will be given a candidate's resume text and a target job description (JD).
Perform a detailed, evidence-based comparison and produce a structured report
in the exact section order below, using Markdown headers and bullet points.

1. Match Overview
   - Give an overall fit score out of 100 between the resume and the JD.
   - Summarize in 2-3 sentences how well the candidate aligns with the role.

2. Strengths
   - List specific skills, experiences, or achievements in the resume that
     directly match the JD's requirements.
   - Reference specific lines/phrases from the resume as evidence.

3. Weaknesses / Gaps
   - Identify required skills, tools, certifications, or experience levels
     from the JD that are missing or weakly represented in the resume.
   - Flag formatting, clarity, or ATS-compatibility issues (missing keywords,
     poor structure, unquantified achievements).

4. Impact vs. Activity Check
   - Go through the resume's bullet points and flag ones that describe pure
     activity ("responsible for", "assisted with", "managed") without a
     measurable outcome.
   - For each flagged bullet, propose a rewritten version that surfaces a
     concrete result or metric (use realistic placeholders like "[X%]" or
     "[$Y]" if no real number is available, and say clearly it's a
     placeholder for the user to fill in).

5. Dual Reviewer Simulation
   - Pass A - ATS Parser: simulate a strict keyword/field-matching ATS pass.
     List exact keywords/phrases from the JD that are missing verbatim from
     the resume, and note any formatting choices likely to break parsing
     (tables, columns, images, non-standard headers).
   - Pass B - Human 7-Second Skim: simulate a recruiter skimming the resume
     for ~7 seconds. State what would actually register in that skim, and
     what important, relevant content is currently buried or easy to miss.

6. Skill Gap & Learning Recommendations
   - List the specific skills/tools/certifications the candidate should learn,
     prioritized as Must-have vs. Nice-to-have.
   - For each Must-have skill, also give one concrete "Portfolio-Proof"
     project idea the candidate could build or complete to demonstrate the
     skill directly on their resume (something more convincing than a
     certificate alone).

7. Actionable Rewrite Suggestions
   - Suggest 3-5 specific edits (rephrasing, added metrics, reordering) to
     better align the resume with the JD.

8. Final Verdict
   - A short, honest summary: is this resume ready to submit? If not, what is
     the single highest-priority fix?

Tone & Style:
- Be honest and constructive, not just encouraging.
- Use clear Markdown headers (##) and bullet points for readability.
- Avoid generic advice; tie every point back to specific evidence from the
  resume or JD.

Constraints:
- Do not fabricate skills or experience the candidate doesn't have.
- If the JD is vague or missing key details, state the assumptions you made.
- If you reference external learning resources, describe them generally
  (name/provider) rather than inventing specific URLs you cannot verify.
"""


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> str:
    """Extract raw text from an uploaded PDF file path."""
    if pdf_file is None:
        return ""
    reader = PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text).strip()


def build_user_message(resume_text: str, jd_text: str) -> str:
    return (
        "=== RESUME TEXT (extracted from PDF) ===\n"
        f"{resume_text}\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{jd_text}\n\n"
        "Please produce the full structured report as instructed."
    )


def analyze_resume(api_key: str, pdf_file, jd_text: str, progress=gr.Progress()):
    """Main callback: extract PDF text, call the model, return Markdown."""
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your OpenRouter API key before running the analysis."

    if pdf_file is None:
        return "⚠️ Please upload a resume/CV in PDF format."

    if not jd_text or not jd_text.strip():
        return "⚠️ Please paste the target job description."

    progress(0.1, desc="Extracting text from PDF...")
    try:
        resume_text = extract_text_from_pdf(pdf_file)
    except Exception:
        return f"❌ Failed to read the PDF file.\n\n```\n{traceback.format_exc()}\n```"

    if not resume_text:
        return "⚠️ Couldn't extract any text from that PDF. It may be a scanned/image-only file."

    progress(0.35, desc="Sending request to the model...")
    try:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key.strip(),
            timeout=90.0,       # give slow connections more room before giving up
            max_retries=3,      # auto-retry transient network/connect errors
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(resume_text, jd_text)},
            ],
            extra_body={"reasoning": {"enabled": True}},
        )

        progress(0.9, desc="Formatting report...")
        message = response.choices[0].message
        content = message.content or ""

        if not content.strip():
            return "⚠️ The model returned an empty response. Try again, or check your API key/credits."

        return content

    except APITimeoutError:
        return (
            "❌ **Connection timed out** reaching OpenRouter (openrouter.ai).\n\n"
            "This is a network issue, not a bug in the analysis logic. Things to check:\n"
            "- Your internet connection is stable right now (try loading https://openrouter.ai in a browser)\n"
            "- A VPN, proxy, firewall, or antivirus isn't blocking/slowing outbound HTTPS traffic\n"
            "- If you're on a corporate/school network, it may be blocking the API host\n"
            "- Try again in a minute — OpenRouter's free-tier endpoints can be briefly overloaded\n\n"
            "The app already retries 3 times with a 90-second timeout, so if it still times out, "
            "it's very likely your network path to openrouter.ai rather than your code."
        )
    except APIConnectionError:
        return (
            "❌ **Couldn't connect** to OpenRouter (openrouter.ai) at all.\n\n"
            "Check that you have internet access, that no firewall/VPN is blocking the request, "
            "and that `https://openrouter.ai` loads normally in your browser. Then try again."
        )
    except AuthenticationError:
        return (
            "❌ **Authentication failed.** Your OpenRouter API key looks invalid or expired. "
            "Double-check you copied the full key (starts with `sk-or-v1-...`) with no extra spaces."
        )
    except RateLimitError:
        return (
            "❌ **Rate limit / no credits.** You may have hit OpenRouter's free-tier rate limit, "
            "or run out of free credits for this model. Wait a bit and try again, or check your "
            "OpenRouter dashboard for usage limits."
        )
    except Exception:
        return (
            "❌ Something unexpected went wrong while calling the model.\n\n"
            f"Details:\n```\n{traceback.format_exc()}\n```"
        )


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------

with gr.Blocks(title="Resume / CV Analyzer") as demo:
    gr.Markdown(
        """
        # 📄 Resume / CV Analyzer
        Upload your resume (PDF) and paste a job description. The app compares
        them and returns a detailed, structured review — strengths, gaps,
        an ATS + recruiter-skim simulation, and a prioritized skill-building plan.

        **Your OpenRouter API key is used only for this session and is never stored.**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="OpenRouter API Key",
                placeholder="sk-or-v1-...",
                type="password",
            )
            pdf_input = gr.File(
                label="Upload Resume (PDF)",
                file_types=[".pdf"],
                type="filepath",
            )
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the full job description here...",
                lines=14,
            )
            analyze_btn = gr.Button("Analyze Resume", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Analysis Report", value="Your report will appear here.")

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[api_key_input, pdf_input, jd_input],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
