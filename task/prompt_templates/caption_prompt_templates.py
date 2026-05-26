"""
Prompt modules for 3D scene caption generation.

Each module category contains multiple phrasings. During prompt assembly,
one phrasing is randomly sampled per category, with per-category dropout
for diversity.

Module categories:
    role       — analyst persona
    task       — task description (100-200 word spatial caption)
    subject    — object analysis focus
    technical  — photographic / composition analysis
    text       — text transcription instructions
    constraint — objectivity constraints
    style      — output formatting style
"""

# ─── Module pools ─────────────────────────────────────────────────

caption_role_modules = [
    "You are a professional image analyst specializing in detailed visual spatial analysis.",
    "Acting as a computer vision specialist, you excel at systematic spatial reasoning.",
    "You're an expert visual content curator with extensive experience in spatial reasoning.",
    "As a digital media researcher, you focus on comprehensive spatial relationship analysis.",
    "You are a trained photographer's assistant skilled in technical spatial analysis.",
    "Working as a visual AI trainer, you specialize in creating precise spatial descriptions.",
    "You're a museum cataloger expert in detailed artifact and scene documentation.",
    "As a forensic image analyst, you provide meticulous spatial evidence documentation.",
]

caption_task_modules = [
    "Create comprehensive spatial relationship descriptions that capture every observable detail in 100-200 words.",
    "Generate systematic visual documentation focusing on spatial relationships of object positions in 100-200 words.",
    "Develop detailed scene inventories that catalog all visible elements and their spatial relationships in 100-200 words.",
    "Produce structured spatial layout analysis report containing both descriptive text and technical metadata in 100-200 words.",
    "Construct thorough image assessments covering spatial, temporal, and contextual elements in 100-200 words.",
]

caption_subject_modules = [
    "Focus on primary subjects including their positioning, appearance, actions, and interactions.",
    "Document all visible entities with emphasis on spatial relationships, sizes, and orientations.",
    "Catalog main objects, describing their materials, conditions, colors, and relative positions within the frame.",
    "Analyze foreground subjects noting their scale, orientation, distinctive features and relative positions.",
    "Record all visual elements with attention to textures, patterns, brands, spatial relationships.",
    "Document key subjects through systematic observation of shapes, colors, sizes, and spatial arrangements.",
]

caption_technical_modules = [
    "Note image composition including camera angle, framing style, depth of field, and any post-processing effects.",
    "Record technical aspects such as lighting conditions, exposure levels, motion blur, and image quality factors.",
    "Document photographic elements including perspective, focus distribution, color grading, and artistic techniques.",
    "Analyze visual presentation covering composition rules, lighting setup, clarity levels, and stylistic choices.",
    "Examine technical execution including shooting angle, depth relationships, exposure balance, and image processing.",
    "Record production qualities such as focus accuracy, lighting direction, color temperature, and editing effects.",
    "Detail photographic characteristics including framing decisions, depth control, lighting quality, and post-production.",
    "Document technical specifications covering composition style, exposure settings, focus distribution, and image treatment.",
]

caption_text_modules = [
    "Transcribe all visible text exactly as it appears, including signs, labels, screens, and printed materials.",
    "Copy any readable text word-for-word, preserving capitalization, punctuation, and formatting.",
    "Document textual elements including signage, captions, digital displays, and written materials with precise accuracy.",
    "Record all text content verbatim, noting position, language, and context of written information.",
    "Capture text elements completely, including partial words, numbers, symbols, and special characters.",
    "Transcribe visible writing, printing, or digital text maintaining original spelling and formatting.",
    "Document readable content including brand names, instructions, labels, and informational text.",
    "Record textual information exactly, preserving case sensitivity and punctuation marks.",
]

caption_constraint_modules = [
    "Maintain objective documentation without subjective interpretations, emotional language, or aesthetic judgments.",
    "Focus exclusively on observable facts, avoiding speculation, mood descriptions, or quality assessments.",
    "Provide factual descriptions only, excluding atmospheric interpretations, artistic evaluations, or inferential content.",
    "Document visible elements without emotional characterizations, beauty assessments, or subjective analysis.",
    "Restrict content to concrete observations, avoiding interpretive language, feeling descriptions, or evaluative terms.",
    "Keep descriptions factual and neutral, excluding mood setting, artistic critique, or speculative content.",
    "Maintain clinical objectivity without aesthetic commentary, emotional interpretation, or subjective evaluation.",
    "Focus on measurable, visible facts while avoiding interpretive analysis, mood descriptions, or quality judgments.",
]

caption_style_modules = [
    "Present information in flowing, natural language that reads smoothly while maintaining technical precision.",
    "Structure content systematically with clear organization and logical progression through visual elements.",
    "Deliver comprehensive analysis using detailed, descriptive language that captures nuanced observations.",
    "Format descriptions as detailed inventories with systematic coverage of all visual components.",
    "Organize information hierarchically, moving from prominent elements to background details.",
    "Present findings in narrative form that guides readers through the visual scene methodically.",
    "Structure analysis as comprehensive reports covering all aspects of visual content systematically.",
    "Deliver content in layered format, building from basic observations to complex scene relationships.",
]


# ─── Assembled dict (imported by CaptionGenerator) ───────────────

CAPTION_MODULES = {
    "role": caption_role_modules,
    "task": caption_task_modules,
    "subject": caption_subject_modules,
    "technical": caption_technical_modules,
    "text": caption_text_modules,
    "constraint": caption_constraint_modules,
    "style": caption_style_modules,
}

CAPTION_DATASET_KEYS = frozenset({"role", "task"})
CAPTION_DATASET_MODULES = {
    "role": caption_role_modules,
    "task": caption_task_modules,
}
CAPTION_API_MODULES = CAPTION_MODULES

CAPTION_DEFAULT_DROPOUT = {
    "role": 0.1,
    "task": 0.0,
    "subject": 0.2,
    "technical": 0.5,
    "text": 0.0,
    "constraint": 0.4,
    "style": 0.2,
}


from ..annotation.core.structured_prompt_template import AnswerInstructionProfile
from .register_structured import register_oe


def register_structured_caption_templates() -> None:
    register_oe(
        "caption.open_ended",
        caption_task_modules,
        [""],
        introduction=caption_role_modules,
        answer_profiles={
            "generative": AnswerInstructionProfile("generative", answer_templates=[""]),
        },
        instruction_types=["generative"],
    )


register_structured_caption_templates()
