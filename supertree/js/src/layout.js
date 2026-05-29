export function createTreeLayoutHelpers(config) {
  const {
    treeDataRoot,
    treeLeafSpacing,
    treeDepthSpacing,
    treeLevelSpacing,
    xMultiplayer,
    yMultiplayer,
  } = config;

  let nodeIdCounter = 0;

  function convertData(node) {
    node.id = nodeIdCounter++;
    if (node.is_leaf) {
      return { children: [], ...node };
    }

    return {
      children: [
        convertData(node.left_node),
        convertData(node.right_node),
      ],
      ...node,
    };
  }

  const treeDataConverted = convertData(treeDataRoot);

  function createHierarchyFromData() {
    const hierarchy = d3.hierarchy(treeDataConverted);
    hierarchy.each(function(node) {
      node.id = node.data.id;
    });
    return hierarchy;
  }

  function computeStableLayout(rootData) {
    const fullHierarchy = d3.hierarchy(rootData);
    let fullMaxDepth = 0;
    let fullLeafs = 0;

    fullHierarchy.eachAfter(function(node) {
      fullMaxDepth = Math.max(fullMaxDepth, node.depth);
      if (!node.children || node.children.length === 0) {
        fullLeafs++;
        node.leafCount = 1;
      } else {
        node.leafCount = node.children.reduce(
          (totalLeafs, child) => totalLeafs + child.leafCount,
          0,
        );
      }
    });

    const leafCompression = Math.log2(Math.max(fullLeafs, 2));
    const depthCompression = Math.log2(Math.max(fullMaxDepth + 1, 2));
    const effectiveLeafSpacing = Math.max(
      treeLeafSpacing * 0.58,
      treeLeafSpacing - leafCompression * 18,
    );
    const effectiveLevelSpacing = Math.max(
      treeLevelSpacing * 0.9,
      treeLevelSpacing - depthCompression * 4,
    );
    const layoutWidth = Math.max(fullLeafs - 1, 1) * effectiveLeafSpacing + treeLeafSpacing;
    const layoutHeight = Math.max(fullMaxDepth, 1) * effectiveLevelSpacing + treeDepthSpacing * 0.4;
    const laidOutHierarchy = d3
      .tree()
      .size([layoutWidth, layoutHeight])
      .separation(function(a, b) {
        const sameParentFactor = a.parent === b.parent ? 1 : 1.2;
        const leafSpan = Math.max((a.leafCount + b.leafCount) / 2, 1);
        const compressionFactor = 1 / Math.pow(leafSpan, 0.18);

        return sameParentFactor * compressionFactor;
      })(fullHierarchy);
    const stableNodePositions = new Map();

    laidOutHierarchy.descendants().forEach(function(node) {
      stableNodePositions.set(node.data.id, {
        x: node.x,
        y: node.depth * effectiveLevelSpacing,
      });
    });

    return {
      effectiveLeafSpacing,
      effectiveLevelSpacing,
      layoutWidth,
      layoutHeight,
      stableNodePositions,
    };
  }

  function applyStableLayout(hierarchyRoot, stableLayout) {
    hierarchyRoot.each(function(node) {
      node.id = node.data.id;
      const stablePosition = stableLayout.stableNodePositions.get(node.data.id);

      node.x = stablePosition.x;
      node.y = stablePosition.y;

      if (!node.hasOwnProperty("cx")) {
        node.cx = node.x * xMultiplayer;
      }
      if (!node.hasOwnProperty("cy")) {
        node.cy = node.y * yMultiplayer;
      }
      if (node.x0 === undefined) {
        node.x0 = node.x;
      }
      if (node.y0 === undefined) {
        node.y0 = node.y;
      }
    });
  }

  function showDepth(node, currentDepth, depth) {
    currentDepth++;
    if (currentDepth < depth) {
      if (node._children) {
        node.children = node._children;
        node._children = null;
      }
      if (node.children) {
        node.children.forEach(function(child) {
          showDepth(child, currentDepth, depth);
        });
      }
    } else if (currentDepth >= depth) {
      if (node.children) {
        node._children = node.children;
        node.children = null;
      }
      if (node._children) {
        node._children.forEach(function(child) {
          showDepth(child, currentDepth, depth);
        });
      }
    }
  }

  function collapse(node, depth, maxDepth) {
    depth++;
    if (node.children && depth == maxDepth) {
      node._children = node.children;
      node.children = null;
    } else if (node.children) {
      node.children.forEach(function(child) {
        collapse(child, depth, maxDepth);
      });
    }
  }

  function getTreeDepth(node, depth) {
    let maxDepth = depth + 1;
    if (node.children) {
      node.children.forEach(function(child) {
        maxDepth = Math.max(maxDepth, getTreeDepth(child, depth + 1));
      });
    }
    if (node._children) {
      node._children.forEach(function(child) {
        maxDepth = Math.max(maxDepth, getTreeDepth(child, depth + 1));
      });
    }
    return maxDepth;
  }

  return {
    treeDataConverted,
    createHierarchyFromData,
    computeStableLayout,
    applyStableLayout,
    showDepth,
    collapse,
    getTreeDepth,
  };
}
