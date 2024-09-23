import os
from string import Template
import random
from pathlib import Path


def get_d3_html(tree_data,start_depth, licence_key="KEY"):
    current_dir = Path(__file__).parent.resolve()
    css_text = ""
    with open(os.path.join(current_dir, "js", "style.css"), encoding='utf-8') as fin:
        css_text = fin.read()

    js_template = ""
    with open(os.path.join(current_dir, "js", "supertree.min.js"), encoding='utf-8') as fin:
        js_template = fin.read()


    myID = str(random.randint(1, 100000))
    js_text = js_template.replace('"$treetemplate"', tree_data)
    js_text = js_text.replace("treeID", myID)
    js_text = js_text.replace("st-licence-KEY", licence_key)
    js_text = js_text.replace('"$depth"',str(start_depth))
    html_template = Template(
        """
    <style > $css_text </style>
    <html>

    <div id="my-window" class="st-container">
        <div id="st-info-div-$treeID" class="st-info-div"></div>
        <div id="toolbar-$treeID" class="st-body-toolbar-div"></div>
        <div id="graph-div-$treeID" class="st-body-tree-div-$treeID"></div>
      <div id="st-side-panel-$treeID" class="st-side-panel">
            <span id="st-close-button-$treeID" class="st-close-button">&times;</span>
        <div>
    </div>
    </html>
    <script > $js_text </script>
    """
    )

    super_tree = html_template.substitute(
        {"css_text": css_text, "js_text": js_text, "treeID": myID}
    )

    return super_tree
