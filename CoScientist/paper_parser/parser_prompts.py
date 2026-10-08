cls_prompt = """
Act as a scientific image classifier. Analyze the input image from an academic paper and determine if it contains meaningful scientific information relevant to the article's content.

**Meaningful images (True):**
- Research diagrams, charts, or graphs
- Experimental photographs or microscopy images
- Data tables with substantive content
- Mathematical formulas/equations
- Technical schematics or flowcharts
- Biological/chemical structures
- Statistical visualizations

**Non-meaningful images (False):**
- Journal/publisher logos
- Decorative icons (social media, print, download etc.)
- Banner advertisements
- Author photos or institutional emblems
- Pure decorative elements
- Copyright watermarks
- Navigation buttons

**Output Rules:**
1. Return only single-word verdict: 'True' for meaningful images, 'False' for non-meaningful
2. Prioritize false negatives over false positives
3. Assume small size (<150px) suggests non-meaningful content
4. Ignore textual content within logos/icons

**Classification standard:**
An image is only 'True' if it conveys scientific data/results/methods essential for understanding the paper's research.

Now classify this image:"""

table_extraction_prompt = """
You are a scientific document analysis expert. Strictly follow these steps when processing the chemistry paper image:

1. Detect all table-like structures in the image:
   - If exactly ONE complete table exists (fully visible borders, headers, and data cells with no cropping/obscuring)
     → Extract tabular data and convert to HTML using <table>, <tr>, <th>, <td> tags
     → Return ONLY the HTML code

2. In ALL other cases return EXACTLY:
   'No table'
   This includes when:
   - Table contains drawn chemical structures that could not be converted to text
   - No table is detected
   - Multiple tables exist
   - Table is incomplete/cropped/obscured
   - Table headers are missing
   - Part of table extends beyond image boundaries

3. Formatting rules:
   - Never add explanations, comments or markdown
   - Don't process non-tabular content
   - Skip text recognition outside tables
   - Omit confidence indicators
   - Output either pure HTML or exact string 'No table'

Critical: Your response must contain ONLY one of two options:
Option A: <table>...</table> (for single valid table)
Option B: No table (all other cases)
"""