export function createViewportController(config) {
  const {
    modal,
    btn,
    span,
    myid,
    rectWidth,
    rectHeight,
    nodeTreeMargin,
    durations,
    stableLayout,
    zoom,
    getTreeRoot,
    getTreeSVG,
    onControlsLockedChange,
  } = config;
  const fallbackDuration = durations.fit;

  let isLocked = false;
  let lastViewportTransform = null;

  function setControlsLocked(locked) {
    isLocked = locked;
    onControlsLockedChange(locked);
  }

  function getIsLocked() {
    return isLocked;
  }

  function runAfterRender(callback) {
    if (typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(function() {
        window.requestAnimationFrame(callback);
      });
      return;
    }

    setTimeout(callback, 0);
  }

  function isModalVisible() {
    return modal.style.display === "block";
  }

  function getViewportContext() {
    if (isModalVisible()) {
      const divElement = document.getElementsByClassName(`st-tree-div-${myid}`)[0];
      return {
        divWidth: divElement.clientWidth,
        divHeight: divElement.clientHeight,
        svgElement: d3.select(divElement).select("svg#mySVG-treeID"),
      };
    }

    const divElement = document.getElementsByClassName("st-body-tree-div-treeID")[0];
    return {
      divWidth: divElement.clientWidth,
      divHeight: divElement.clientHeight,
      svgElement: d3.select(divElement).select("svg#mySVG-treeID"),
    };
  }

  function getRenderedTreeBounds() {
    const treeNode = getTreeSVG().node();

    if (!treeNode) {
      return null;
    }

    const treeBounds = treeNode.getBBox();

    if (!Number.isFinite(treeBounds.width) || !Number.isFinite(treeBounds.height)) {
      return null;
    }

    const horizontalPadding = rectWidth / 2 + 40;
    const verticalPadding = rectHeight / 2 + 40;

    return {
      minX: treeBounds.x - horizontalPadding,
      maxX: treeBounds.x + treeBounds.width + horizontalPadding,
      minY: treeBounds.y - verticalPadding,
      maxY: treeBounds.y + treeBounds.height + verticalPadding,
      width: Math.max(treeBounds.width + horizontalPadding * 2, 1),
      height: Math.max(treeBounds.height + verticalPadding * 2, 1),
    };
  }

  function getVisibleTreeBounds(options = {}) {
    const visibleNodes = getTreeRoot().descendants();

    if (
      options.fallbackToFullForSingleRoot &&
      visibleNodes.length === 1 &&
      !visibleNodes[0].parent
    ) {
      return getFullTreeBounds();
    }

    const renderedBounds = getRenderedTreeBounds();

    if (renderedBounds !== null) {
      return renderedBounds;
    }

    const horizontalPadding = rectWidth / 2 + 40;
    const verticalPadding = rectHeight / 2 + 40;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    visibleNodes.forEach(function(node) {
      minX = Math.min(minX, node.x - horizontalPadding);
      maxX = Math.max(maxX, node.x + horizontalPadding);
      minY = Math.min(minY, node.y - verticalPadding);
      maxY = Math.max(maxY, node.y + verticalPadding);
    });

    return {
      minX,
      maxX,
      minY,
      maxY,
      width: Math.max(maxX - minX, 1),
      height: Math.max(maxY - minY, 1),
    };
  }

  function getFullTreeBounds() {
    const horizontalPadding = rectWidth / 2 + 40;
    const verticalPadding = rectHeight / 2 + 40;
    const positions = Array.from(stableLayout.stableNodePositions.values());

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    positions.forEach(function(position) {
      minX = Math.min(minX, position.x - horizontalPadding);
      maxX = Math.max(maxX, position.x + horizontalPadding);
      minY = Math.min(minY, position.y - verticalPadding);
      maxY = Math.max(maxY, position.y + verticalPadding);
    });

    return {
      minX,
      maxX,
      minY,
      maxY,
      width: Math.max(maxX - minX, 1),
      height: Math.max(maxY - minY, 1),
    };
  }

  function getFitTransform(mode = "visible", options = {}) {
    const { divWidth, divHeight } = getViewportContext();
    const fitBounds = mode === "full" ? getFullTreeBounds() : getVisibleTreeBounds(options);
    const topViewportPadding = 20;
    const availableWidth = Math.max(divWidth - nodeTreeMargin.left - nodeTreeMargin.right, 1);
    const availableHeight = Math.max(divHeight - nodeTreeMargin.top - nodeTreeMargin.bottom, 1);
    let currentScale = Math.min(
      availableWidth / fitBounds.width,
      availableHeight / fitBounds.height,
    );
    currentScale = Math.min(Math.max(0.05, currentScale), 2);

    const centeredX = (fitBounds.minX + fitBounds.maxX) / 2;
    return d3.zoomIdentity
      .translate(
        -centeredX * currentScale + divWidth / 2,
        -fitBounds.minY * currentScale + topViewportPadding,
      )
      .scale(currentScale);
  }

  function resetZoom(mode = "visible", transitionDuration = fallbackDuration, options = {}) {
    const { svgElement } = getViewportContext();
    const rootTransform = getFitTransform(mode, options);

    svgElement.interrupt();

    if (transitionDuration <= 0) {
      svgElement.call(zoom.transform, rootTransform);
      return;
    }

    svgElement
      .transition()
      .duration(transitionDuration)
      .call(zoom.transform, rootTransform);
  }

  function rememberViewport() {
    const currentTransform = d3.zoomTransform(getTreeSVG().node());
    lastViewportTransform = currentTransform;
  }

  function restoreViewport(transitionDuration = fallbackDuration) {
    if (!lastViewportTransform) {
      resetZoom("visible", transitionDuration);
      return;
    }

    const { svgElement } = getViewportContext();
    if (transitionDuration <= 0) {
      svgElement.call(zoom.transform, lastViewportTransform);
      return;
    }

    svgElement
      .transition()
      .duration(transitionDuration)
      .call(zoom.transform, lastViewportTransform);
  }

  function finishActionAfter(delay) {
    setTimeout(function() {
      setControlsLocked(false);
    }, delay);
  }

  function applyViewportPolicy(actionType) {
    switch (actionType) {
      case "initial":
        lastViewportTransform = null;
        runAfterRender(function() {
          resetZoom("visible", durations.fit, { fallbackToFullForSingleRoot: false });
          rememberViewport();
        });
        return;
      case "root-toggle":
        runAfterRender(function() {
          resetZoom("full", durations.rootFit);
          rememberViewport();
        });
        finishActionAfter(durations.rootFit);
        return;
      case "depth-change":
      case "sample-path":
      case "fit-visible":
        runAfterRender(function() {
          resetZoom("visible", durations.fit, { fallbackToFullForSingleRoot: true });
          rememberViewport();
        });
        if (actionType === "depth-change" || actionType === "sample-path") {
          finishActionAfter(durations.fit);
        }
        return;
      case "fit-full":
        runAfterRender(function() {
          resetZoom("full", durations.fit);
          rememberViewport();
        });
        return;
      case "modal-open":
        rememberViewport();
        runAfterRender(function() {
          restoreViewport(durations.modal);
        });
        return;
      case "modal-close":
        runAfterRender(function() {
          restoreViewport(durations.modal);
        });
        return;
      case "inner-toggle":
        runAfterRender(function() {
          rememberViewport();
        });
        finishActionAfter(durations.toggle);
        return;
      default:
        return;
    }
  }

  span.onclick = function() {
    modal.style.display = "none";
    applyViewportPolicy("modal-close");
  };

  modal.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
      applyViewportPolicy("modal-close");
    }
  };

  btn.onclick = function() {
    applyViewportPolicy("modal-open");
    modal.style.display = "block";
  };

  return {
    getIsLocked,
    setControlsLocked,
    applyViewportPolicy,
    resetZoom,
    rememberViewport,
  };
}
