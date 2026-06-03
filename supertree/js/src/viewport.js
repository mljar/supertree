export function createViewportController(config) {
  const {
    instanceKey = "default",
    treeType = "",
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
    getAllTreeNodes,
    getTreeSVG,
    getNodeBounds,
    onControlsLockedChange,
  } = config;
  const fallbackDuration = durations.fit;

  let isLocked = false;
  let lastViewportTransform = null;
  let resizeTimer = null;

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

  function runAfterRenderSettled(callback, delay = 30) {
    runAfterRender(function() {
      setTimeout(callback, delay);
    });
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

  function getNodeVisualBounds(node, fallbackPosition = null) {
    const position = fallbackPosition || node;
    const bounds = typeof getNodeBounds === "function"
      ? getNodeBounds(node)
      : null;

    if (!bounds) {
      const horizontalPadding = rectWidth / 2 + 40;
      const verticalPadding = rectHeight / 2 + 40;
      return {
        minX: position.x - horizontalPadding,
        maxX: position.x + horizontalPadding,
        minY: position.y - verticalPadding,
        maxY: position.y + verticalPadding,
      };
    }

    return {
      minX: position.x + bounds.minX,
      maxX: position.x + bounds.maxX,
      minY: position.y + bounds.minY,
      maxY: position.y + bounds.maxY,
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

    if (options.useRenderedBounds !== false && renderedBounds !== null) {
      return renderedBounds;
    }

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    visibleNodes.forEach(function(node) {
      const nodeBounds = getNodeVisualBounds(node);
      minX = Math.min(minX, nodeBounds.minX);
      maxX = Math.max(maxX, nodeBounds.maxX);
      minY = Math.min(minY, nodeBounds.minY);
      maxY = Math.max(maxY, nodeBounds.maxY);
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
    const allNodes = typeof getAllTreeNodes === "function"
      ? getAllTreeNodes()
      : getTreeRoot().descendants();

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    allNodes.forEach(function(node) {
      const stablePosition = stableLayout.stableNodePositions.get(node.data.id);
      const nodeBounds = getNodeVisualBounds(node, stablePosition);
      minX = Math.min(minX, nodeBounds.minX);
      maxX = Math.max(maxX, nodeBounds.maxX);
      minY = Math.min(minY, nodeBounds.minY);
      maxY = Math.max(maxY, nodeBounds.maxY);
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
    const maxScale = options.maxScale ?? 2;
    const availableWidth = Math.max(divWidth - nodeTreeMargin.left - nodeTreeMargin.right, 1);
    const availableHeight = Math.max(divHeight - nodeTreeMargin.top - nodeTreeMargin.bottom, 1);
    let currentScale = Math.min(
      availableWidth / fitBounds.width,
      availableHeight / fitBounds.height,
    );
    currentScale = Math.min(Math.max(0.05, currentScale), maxScale);

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
      return rootTransform;
    }

    svgElement
      .transition()
      .duration(transitionDuration)
      .call(zoom.transform, rootTransform);
    return rootTransform;
  }

  function rememberViewport(transform = null) {
    if (transform) {
      lastViewportTransform = transform;
      return;
    }

    const treeNode = getTreeSVG().node();
    if (!treeNode) {
      return;
    }

    lastViewportTransform = d3.zoomTransform(treeNode);
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

  function handleViewportResize() {
    if (isLocked) {
      return;
    }

    if (resizeTimer) {
      clearTimeout(resizeTimer);
    }

    resizeTimer = setTimeout(function() {
      resizeTimer = null;
      runAfterRenderSettled(function() {
        rememberViewport(resetZoom("visible", 160, {
          fallbackToFullForSingleRoot: true,
          useRenderedBounds: true,
        }));
      }, 40);
    }, 120);
  }

  function applyViewportPolicy(actionType, options = {}) {
    switch (actionType) {
      case "initial":
        lastViewportTransform = null;
        runAfterRenderSettled(function() {
          rememberViewport(resetZoom("visible", 0, {
            fallbackToFullForSingleRoot: false,
            maxScale: 3.5,
            useRenderedBounds: true,
          }));
          runAfterRenderSettled(function() {
            rememberViewport(resetZoom("visible", 0, {
              fallbackToFullForSingleRoot: false,
              maxScale: 3.5,
              useRenderedBounds: true,
            }));
          }, 140);
        }, 60);
        return;
      case "root-toggle":
        runAfterRenderSettled(function() {
          rememberViewport(resetZoom("visible", durations.rootFit, {
            fallbackToFullForSingleRoot: false,
            useRenderedBounds: true,
          }));
        }, 20);
        finishActionAfter(durations.rootFit);
        return;
      case "depth-change":
      case "sample-path":
        runAfterRenderSettled(function() {
          rememberViewport(
            resetZoom("visible", durations.fit, {
              fallbackToFullForSingleRoot: actionType === "sample-path",
              useRenderedBounds: actionType === "sample-path",
            }),
          );
          runAfterRenderSettled(function() {
            rememberViewport(
              resetZoom("visible", 0, {
                fallbackToFullForSingleRoot: actionType === "sample-path",
                useRenderedBounds: actionType === "sample-path",
              }),
            );
          }, 140);
        }, 40);
        if (actionType === "depth-change" || actionType === "sample-path") {
          finishActionAfter(durations.fit + 180);
        }
        return;
      case "fit-visible":
        runAfterRenderSettled(function() {
          rememberViewport(resetZoom("visible", Math.min(durations.fit, 220), {
            fallbackToFullForSingleRoot: true,
            useRenderedBounds: true,
          }));
        });
        return;
      case "fit-full":
        runAfterRender(function() {
          rememberViewport(resetZoom("full", durations.fit));
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
        runAfterRenderSettled(function() {
          const useRenderedBounds = treeType === "regression";
          rememberViewport(resetZoom("visible", Math.min(durations.fit, 220), {
            fallbackToFullForSingleRoot: true,
            useRenderedBounds,
          }));
          runAfterRenderSettled(function() {
            rememberViewport(resetZoom("visible", 0, {
              fallbackToFullForSingleRoot: true,
              useRenderedBounds,
            }));
          }, 140);
          finishActionAfter(Math.min(durations.fit, 220) + 140);
        }, durations.toggle);
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

  if (btn) {
    btn.onclick = function() {
      applyViewportPolicy("modal-open");
      modal.style.display = "block";
    };
  }

  if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
    if (!window.__supertreeViewportResizeHandlers) {
      window.__supertreeViewportResizeHandlers = {};
    }
    const previousResizeHandler = window.__supertreeViewportResizeHandlers[instanceKey];
    if (previousResizeHandler) {
      window.removeEventListener("resize", previousResizeHandler);
    }
    window.__supertreeViewportResizeHandlers[instanceKey] = handleViewportResize;
    window.addEventListener("resize", handleViewportResize);
  }

  return {
    getIsLocked,
    setControlsLocked,
    applyViewportPolicy,
    resetZoom,
    rememberViewport,
    getVisibleTreeBounds,
  };
}
