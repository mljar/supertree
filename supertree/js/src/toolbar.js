export function createToolbarGroups({
  isJupyter,
  toolbarSelector,
  windowIconMarkup,
}) {
  const toolbarRoot = d3.select(toolbarSelector);
  const primaryToolbarGroup = toolbarRoot
    .append("div")
    .attr("class", "st-toolbar-group");
  const secondaryToolbarGroup = toolbarRoot
    .append("div")
    .attr("class", "st-toolbar-group");
  const depthToolbarGroup = toolbarRoot
    .append("div")
    .attr("class", "st-toolbar-group");
  const tertiaryToolbarGroup = toolbarRoot
    .append("div")
    .attr("class", "st-toolbar-group");

  const modalButton = primaryToolbarGroup
    .append("button")
    .html(windowIconMarkup)
    .attr("class", "st-option-button")
    .attr("id", "openModalBtn-treeID");

  if (!isJupyter) {
    modalButton.style("display", "none");
  }

  return {
    depthToolbarGroup,
    modalButton,
    primaryToolbarGroup,
    secondaryToolbarGroup,
    tertiaryToolbarGroup,
    toolbarRoot,
  };
}

export function setupDepthControls({
  group,
  initialDepth,
  isLocked,
  maxDepth,
  onDepthChange,
}) {
  let currentDepthValue = Math.max(1, Math.min(initialDepth, maxDepth));

  const depthControl = group
    .append("div")
    .attr("class", "st-depth-control");

  const depthDecreaseButton = depthControl
    .append("button")
    .attr("id", "st-depth-decrease-treeID")
    .attr("class", "st-option-button st-depth-button")
    .attr("type", "button")
    .text("-");

  const depthLabel = depthControl
    .append("div")
    .attr("id", "st-depth-label-treeID")
    .attr("class", "st-depth-label");

  const depthIncreaseButton = depthControl
    .append("button")
    .attr("id", "st-depth-increase-treeID")
    .attr("class", "st-option-button st-depth-button")
    .attr("type", "button")
    .text("+");

  function sync() {
    depthLabel.text(`Depth=${Math.max(currentDepthValue - 1, 0)}`);
    depthDecreaseButton.attr(
      "disabled",
      isLocked() || currentDepthValue <= 1 ? "disabled" : null,
    );
    depthIncreaseButton.attr(
      "disabled",
      isLocked() || currentDepthValue >= maxDepth ? "disabled" : null,
    );
  }

  function setDepthValue(nextDepth) {
    currentDepthValue = Math.max(1, Math.min(nextDepth, maxDepth));
  }

  function applyDepthChange(nextDepth) {
    const clampedDepth = Math.max(1, Math.min(nextDepth, maxDepth));
    if (clampedDepth === currentDepthValue) {
      sync();
      return;
    }

    currentDepthValue = clampedDepth;
    sync();
    onDepthChange(clampedDepth);
  }

  depthDecreaseButton.on("click", function() {
    if (isLocked()) {
      sync();
      return;
    }
    applyDepthChange(currentDepthValue - 1);
  });

  depthIncreaseButton.on("click", function() {
    if (isLocked()) {
      sync();
      return;
    }
    applyDepthChange(currentDepthValue + 1);
  });

  sync();

  return {
    getDepthValue() {
      return currentDepthValue;
    },
    setDepthValue,
    sync,
  };
}
