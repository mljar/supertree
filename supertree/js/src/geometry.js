export function createTreeVisualMetrics(treeType) {
  const dimensions = {
    pieHeight: 100,
    pieWidth: 100,
    histogramWidth: 150,
    histogramHeight: 80,
    rectHeight: 110,
    rectWidth: 195,
  };

  dimensions.scatterplotWidth = dimensions.histogramWidth;
  dimensions.scatterplotHeight = dimensions.histogramHeight;
  dimensions.scatterplotLeafWidth = dimensions.histogramHeight + 10;
  dimensions.scatterplotLeafHeight = dimensions.histogramHeight;

  const spacing = {
    treeLeafSpacing: treeType === "regression" ? 340 : 300,
    treeLevelSpacing: 235,
    treeDepthSpacing: 400,
  };

  const layout = {
    histogramTranslateX: -dimensions.histogramWidth / 2,
    histogramRectX: -(dimensions.scatterplotWidth / 2) - 25,
    classificationLeafRectX: -(dimensions.scatterplotWidth / 2) - 5,
    regressionLeafRectX: -(dimensions.scatterplotLeafWidth / 2) - 10,
    regressionLeafPlotTranslateX: -dimensions.scatterplotLeafWidth / 2 + 15,
    pieCenterX: 10,
    leafLabelX: 15,
    nodeLabelY: dimensions.rectHeight + 15,
    regressionLeafLabelY: dimensions.rectHeight + 5,
  };

  function getViewportBounds(node) {
    if (treeType === "regression") {
      if (node.data && node.data.is_leaf) {
        return {
          minX: -70,
          maxX: 80,
          minY: -14,
          maxY: 162,
        };
      }

      return {
        minX: -100,
        maxX: 98,
        minY: -14,
        maxY: 142,
      };
    }

    if (treeType === "classification") {
      if (node.data && node.data.is_leaf) {
        return {
          minX: -100,
          maxX: 112,
          minY: -14,
          maxY: 210,
        };
      }

      return {
        minX: -100,
        maxX: 98,
        minY: -14,
        maxY: 142,
      };
    }

    return {
      minX: -100,
      maxX: 98,
      minY: -14,
      maxY: 142,
    };
  }

  function getSourceAnchor(base, childBase = null) {
    const cornerInset = 16;
    let anchorX = base.x;

    if (childBase) {
      if (childBase.x < base.x) {
        anchorX = base.x - dimensions.rectWidth / 2 + cornerInset;
      } else if (childBase.x > base.x) {
        anchorX = base.x + dimensions.rectWidth / 2 - cornerInset;
      }
    }

    return {
      x: anchorX,
      y: base.y + dimensions.rectHeight - 10,
    };
  }

  function getTargetAnchor(node, base) {
    if (node.data && node.data.is_leaf) {
      if (treeType === "classification") {
        return {
          x: base.x + 10,
          y: base.y - 3,
        };
      }

      if (treeType === "regression") {
        return {
          x: base.x + 15,
          y: base.y + 4,
        };
      }
    }

    return {
      x: base.x,
      y: base.y - 10,
    };
  }

  return {
    dimensions,
    layout,
    spacing,
    getViewportBounds,
    getSourceAnchor,
    getTargetAnchor,
  };
}
