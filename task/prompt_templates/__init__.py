"""
Auto-import template modules to trigger StructuredTemplateRegistry registrations.
"""
from . import distance_prompt_templates
from . import size_prompt_templates
from . import depth_prompt_templates
from . import position_prompt_templates
from . import threed_grounding_prompt_templates
from . import multiview_object_position_templates
from . import correspondence_prompt_templates
from . import caption_prompt_templates  # noqa: F401
