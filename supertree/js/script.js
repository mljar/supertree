(() => {
  // supertree/js/src/icons.js
  var svgLine = `<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-vector-spline"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M17 3m0 1a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v2a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1z" /><path d="M3 17m0 1a1 1 0 0 1 1 -1h2a1 1 0 0 1 1 1v2a1 1 0 0 1 -1 1h-2a1 1 0 0 1 -1 -1z" /><path d="M17 5c-6.627 0 -12 5.373 -12 12" /></svg>`;
  var svgXAxis = `<svg style="display: inline" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 13v.01" /><path d="M4 9v.01" /><path d="M4 5v.01" /><path d="M17 20l3 -3l-3 -3" /><path d="M4 17h16" /></svg>`;
  var svgYAxis = `<svg style="display: inline" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M11 20h-.01" /><path d="M15 20h-.01" /><path d="M19 20h-.01" /><path d="M4 7l3 -3l3 3" /><path d="M7 20v-16" /></svg>`;
  var svgZoom = `<svg style="display: inline" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M10 10m-7 0a7 7 0 1 0 14 0a7 7 0 1 0 -14 0" /><path d="M21 21l-6 -6" /></svg>`;
  var svgWindow = `<svg style="display: inline" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 8v-2a2 2 0 0 1 2 -2h2" /><path d="M4 16v2a2 2 0 0 0 2 2h2" /><path d="M16 4h2a2 2 0 0 1 2 2v2" /><path d="M16 20h2a2 2 0 0 0 2 -2v-2" /></svg>`;
  var svgDownload = `<svg style="display: inline"  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2" /><path d="M7 11l5 5l5 -5" /><path d="M12 4l0 12" /></svg>`;
  var svgSample = `<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-test-pipe-2"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M15 3v15a3 3 0 0 1 -6 0v-15" /><path d="M9 12h6" /><path d="M8 3h8" /></svg>`;
  var svgFitFull = `<svg style="display: inline" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 9v-5h5" /><path d="M20 9v-5h-5" /><path d="M4 15v5h5" /><path d="M20 15v5h-5" /></svg>`;

  // supertree/js/src/layout.js
  function createTreeLayoutHelpers(config) {
    const {
      treeDataRoot,
      treeLeafSpacing,
      treeDepthSpacing,
      treeLevelSpacing,
      xMultiplayer,
      yMultiplayer
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
          convertData(node.right_node)
        ],
        ...node
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
            0
          );
        }
      });
      const leafCompression = Math.log2(Math.max(fullLeafs, 2));
      const depthCompression = Math.log2(Math.max(fullMaxDepth + 1, 2));
      const effectiveLeafSpacing = Math.max(
        treeLeafSpacing * 0.58,
        treeLeafSpacing - leafCompression * 18
      );
      const effectiveLevelSpacing = Math.max(
        treeLevelSpacing * 0.9,
        treeLevelSpacing - depthCompression * 4
      );
      const layoutWidth = Math.max(fullLeafs - 1, 1) * effectiveLeafSpacing + treeLeafSpacing;
      const layoutHeight = Math.max(fullMaxDepth, 1) * effectiveLevelSpacing + treeDepthSpacing * 0.4;
      const laidOutHierarchy = d3.tree().size([layoutWidth, layoutHeight]).separation(function(a, b) {
        const sameParentFactor = a.parent === b.parent ? 1 : 1.2;
        const leafSpan = Math.max((a.leafCount + b.leafCount) / 2, 1);
        const compressionFactor = 1 / Math.pow(leafSpan, 0.18);
        return sameParentFactor * compressionFactor;
      })(fullHierarchy);
      const stableNodePositions = /* @__PURE__ */ new Map();
      laidOutHierarchy.descendants().forEach(function(node) {
        stableNodePositions.set(node.data.id, {
          x: node.x,
          y: node.depth * effectiveLevelSpacing
        });
      });
      return {
        effectiveLeafSpacing,
        effectiveLevelSpacing,
        layoutWidth,
        layoutHeight,
        stableNodePositions
      };
    }
    function applyStableLayout(hierarchyRoot, stableLayout) {
      const visibleLayoutWidth = computeVisibleLayoutWidth(hierarchyRoot, stableLayout);
      const visibleLaidOutHierarchy = d3.tree().size([visibleLayoutWidth, stableLayout.layoutHeight]).separation(function(a, b) {
        return a.parent === b.parent ? 1 : 1.2;
      })(hierarchyRoot.copy());
      const visibleNodePositions = /* @__PURE__ */ new Map();
      visibleLaidOutHierarchy.descendants().forEach(function(node) {
        visibleNodePositions.set(node.data.id, node.x);
      });
      hierarchyRoot.each(function(node) {
        node.id = node.data.id;
        const stablePosition = stableLayout.stableNodePositions.get(node.data.id);
        const visibleX = visibleNodePositions.get(node.data.id);
        node.x = visibleX !== void 0 ? visibleX : stablePosition.x;
        node.y = stablePosition.y;
        if (!node.hasOwnProperty("cx")) {
          node.cx = node.x * xMultiplayer;
        }
        if (!node.hasOwnProperty("cy")) {
          node.cy = node.y * yMultiplayer;
        }
        if (node.x0 === void 0) {
          node.x0 = node.x;
        }
        if (node.y0 === void 0) {
          node.y0 = node.y;
        }
      });
    }
    function computeVisibleLayoutWidth(hierarchyRoot, stableLayout) {
      let visibleLeafs = 0;
      hierarchyRoot.each(function(node) {
        if (!node.children || node.children.length === 0) {
          visibleLeafs++;
        }
      });
      const normalizedLeafCount = Math.max(visibleLeafs, 1);
      const visibleWidth = Math.max(normalizedLeafCount - 1, 1) * stableLayout.effectiveLeafSpacing + stableLayout.effectiveLeafSpacing;
      return Math.min(Math.max(visibleWidth, stableLayout.effectiveLeafSpacing * 2), stableLayout.layoutWidth);
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
      getTreeDepth
    };
  }

  // supertree/js/src/shared.js
  var DEBUG_LEVEL = "debug";
  var yAxisMargin = 25;
  function stLog(level, obj, message = "Data;") {
    const levels = ["debug", "custom", "info", "warning", "error"];
    if (levels.indexOf(level) >= levels.indexOf(DEBUG_LEVEL)) {
      console.log(`[${level.toUpperCase()}]:`, message, obj);
    }
  }

  // supertree/js/src/node_renderers.js
  function processClassificationNode(treeData, tooltipBody, tooltipModal, globalX, globalXExtent, globalY, globalYExtent, click, histogramWidth, histogramHeight, rectWidth, rectHeight, colors, d) {
    var isSampleExist = false;
    if (!d.data.is_leaf) {
      if (treeData.show_sample != "nodata" && treeData.show_sample != void 0) {
        isSampleExist = true;
      }
      var isSampleExistInThisNode = true;
      if (isSampleExist) {
        isSampleExistInThisNode = true;
      } else {
        isSampleExistInThisNode = false;
      }
      const featureIndex = d.data.feature;
      const uniqueTargets = [...new Set(treeData.data_target)];
      const maxTarget = Math.max(...uniqueTargets);
      let filteredData = Array.from({ length: maxTarget + 1 }, () => []);
      stLog("debug", filteredData, "filtered");
      stLog("debug", treeData.data_target, "data_target");
      treeData.data_feature.forEach((row, index) => {
        const target = treeData.data_target[index];
        filteredData[target].push(row);
      });
      var xExtent2 = [Infinity, -Infinity];
      d.data.start_end_x_axis.forEach((currentElement, index) => {
        if (currentElement[0] != "notexist") {
          filteredData.forEach((twoDimArray, i) => {
            filteredData[i] = twoDimArray.filter(
              (rowArray) => rowArray[index] < currentElement[0]
            );
          });
        }
        if (currentElement[1] != "notexist") {
          filteredData.forEach((twoDimArray, i) => {
            filteredData[i] = twoDimArray.filter(
              (rowArray) => rowArray[index] > currentElement[1]
            );
          });
        }
      });
      if (isSampleExist) {
        d.data.start_end_x_axis.forEach((currentElement, i) => {
          const sampleValue = treeData.show_sample[i];
          if (currentElement[0] != "notexist" && sampleValue > currentElement[0]) {
            isSampleExistInThisNode = false;
          }
          if (currentElement[1] != "notexist" && sampleValue < currentElement[1]) {
            isSampleExistInThisNode = false;
          }
        });
      }
      let featureData = filteredData.map(
        (subArray) => subArray.map((innerArray) => innerArray[featureIndex])
      );
      const removedIndices = [];
      let indicesArray = Array.from(
        { length: featureData.length + removedIndices.length },
        (_, i) => i
      );
      featureData = featureData.map((twoDimArray, index) => {
        const filteredArray = twoDimArray.filter(
          (value) => value !== void 0 && value !== null
        );
        if (filteredArray.length === 0) {
          removedIndices.push(index);
        }
        return filteredArray;
      }).filter((twoDimArray) => twoDimArray.length > 0);
      let removedSet = new Set(removedIndices);
      indicesArray = indicesArray.filter(
        (value) => !removedSet.has(value)
      );
      featureData.map((currentValue) => {
        var tempxExtent = d3.extent(currentValue);
        xExtent2[0] = Math.min(xExtent2[0], tempxExtent[0]);
        xExtent2[1] = Math.max(xExtent2[1], tempxExtent[1]);
      });
      xExtent2[0] = xExtent2[0] - 0.2;
      xExtent2[1] = xExtent2[1] + 0.2;
      if (globalX) {
        xExtent2 = globalXExtent[featureIndex];
      }
      if (xExtent2[0] > d.data.threshold) {
        xExtent2[0] = d.data.threshold - 0.2;
      }
      if (xExtent2[1] < d.data.threshold) {
        xExtent2[1] = d.data.threshold + 0.2;
      }
      if (isSampleExistInThisNode) {
        if (xExtent2[0] > treeData.show_sample[featureIndex]) {
          xExtent2[0] = treeData.show_sample[featureIndex] - 0.2;
        }
        if (xExtent2[1] < treeData.show_sample[featureIndex]) {
          xExtent2[1] = treeData.show_sample[featureIndex] + 0.2;
        }
      }
      const xScale = d3.scaleLinear().domain(xExtent2).range([0, histogramWidth]);
      d3.select(this).append("rect").attr("class", "histogram-background").attr("x", -(histogramWidth / 2) - 25).attr("y", -10).attr("width", rectWidth).attr("height", rectHeight).attr("stroke-width", 1).attr("stroke", "#545454").attr("rx", 10).attr("ry", 10).style("fill", "#ffffff").on("click", click);
      const mousemoveAllData = function(event2, d2) {
        stLog("debug", d2, "mousemoveAllData");
        tooltipBody.html(
          `<b>All Data</b>:<br>Class distribution: ${d2.data.class_distribution}
              <br>Impurity: ${d2.data.impurity}
              <br>Samples: ${d2.data.samples}
              <br>Threshold: ${parseFloat(d2.data.threshold).toFixed(3)}
              <br>Treeclass: ${d2.data.treeclass}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px").style("opacity", 1);
        tooltipModal.html(
          `<b>All Data</b>:<br>Class distribution: ${d2.data.class_distribution}
              <br>Impurity: ${d2.data.impurity}
              <br>Samples: ${d2.data.samples}
              <br>Threshold: ${parseFloat(d2.data.threshold).toFixed(3)}
              <br>Treeclass: ${d2.data.treeclass}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px").style("opacity", 1);
      };
      const mouseleaveAllData = function(event2, d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
      };
      stLog("debug", this, "This");
      d3.select(this).append("text").attr("class", "st-target").attr("x", 0).attr("y", rectHeight + 15).style("text-anchor", "middle").style("font-size", "18px").text(treeData.feature_names[featureIndex]).on("mousemove", mousemoveAllData).on("mouseleave", mouseleaveAllData).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
      const xDomain = xScale.domain();
      const xTickValues = [
        xDomain[0],
        d.data.threshold,
        xDomain[1]
      ];
      d3.select(this).append("g").attr("class", "xAxis").attr(
        "transform",
        `translate(${-histogramWidth / 2}, ${histogramHeight})`
      ).call(
        d3.axisBottom(xScale).tickSize(0).tickPadding(8).tickValues(xTickValues).tickFormat(d3.format(",.1f"))
      ).selectAll(".tick").attr("class", "xAxis-text").style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("fill", "black");
      d3.select(this).selectAll(".domain").style("stroke", "black");
      const yScale = d3.scaleLinear().range([histogramHeight, 0]);
      const yAxis = d3.select(this).append("g");
      const histogram = d3.bin().domain(xScale.domain()).thresholds(xScale.ticks(20));
      const binsData = featureData.map((data) => histogram(data));
      stLog("debug", binsData, "binsData");
      const stackedData = d3.stack().keys(d3.range(featureData.length)).value((d2, key) => d2[key] ? d2[key].length : 0)(
        d3.transpose(binsData)
      );
      var yExtent2 = [0, 0];
      yExtent2[0] = 0;
      yExtent2[1] = d3.max(
        stackedData,
        (d2) => d3.max(d2, (d4) => d4[1])
      );
      globalYExtent[featureIndex][1] = Math.max(
        globalYExtent[featureIndex][1],
        d3.max(stackedData, (d2) => d3.max(d2, (d4) => d4[1]))
      );
      if (globalY) {
        yExtent2 = globalYExtent[featureIndex];
      }
      yScale.domain(yExtent2);
      const yDomain = yScale.domain();
      const yTickValues = [
        yDomain[0],
        yDomain[0] + (yDomain[1] - yDomain[0]) / 2,
        yDomain[1]
      ];
      stLog("debug", yTickValues, "YTICKVALUES");
      if (yTickValues.every((value) => !isNaN(value))) {
        yAxis.call(
          d3.axisRight(yScale).tickSize(0).tickPadding(4).tickValues(yTickValues).tickFormat(d3.format(",.0f"))
        ).attr("class", "yAxis").attr("transform", `translate(${-histogramWidth / 2 - yAxisMargin},0)`).call((d2) => d2.select(".domain").remove()).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("fill", "black");
      }
      d3.select(this).selectAll(".domain").style("stroke", "black");
      const mouseover = function(d2) {
        tooltipModal.style("opacity", 1);
        tooltipBody.style("opacity", 1);
        d3.select(this).style("stroke", "#EF4A60");
      };
      const mouseleave = function(d2) {
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("stroke", "black");
      };
      const mousemove = function(event2, d2) {
        tooltipBody.html(
          `<b>${treeData.target_names[d2.class]}</b>: ${d2[1] - d2[0]}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(
          `<b>${treeData.target_names[d2.class]}</b>: ${d2[1] - d2[0]}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      let nodeToClick = d;
      stLog("debug", stackedData, "stackedData");
      stLog("debug", d.data, "node:");
      d3.select(this).selectAll("g.layer").data(stackedData).enter().append("g").attr("class", "layer").style("fill", (d2, i) => colors[indicesArray[i]]).on("click", function() {
        click(source, nodeToClick);
      }).selectAll("rect.bar").data(
        (d2, i) => d2.map((item) => ({ ...item, class: indicesArray[i] }))
      ).enter().append("rect").attr("class", "bar").attr("x", (d2) => {
        return xScale(d2.data[0].x0);
      }).attr("y", (d2) => {
        if (isNaN(yScale(d2[1]))) {
          return 0;
        } else {
          return yScale(d2[1]);
        }
      }).attr("height", (d2) => {
        if (isNaN(yScale(d2[0]) - yScale(d2[1]))) {
          return 0;
        } else {
          return yScale(d2[0]) - yScale(d2[1]);
        }
      }).attr(
        "width",
        (d2) => xScale(d2.data[0].x1) - xScale(d2.data[0].x0)
      ).attr("transform", `translate(${-histogramWidth / 2},0)`).attr("stroke", "black").on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", mousemove);
      stLog("debug", "abc");
      var threshold = parseFloat(d.data.threshold).toFixed(3);
      d3.select(this).append("line").attr("class", "threshold-line").attr("x1", xScale(threshold)).attr("x2", xScale(threshold)).attr("y1", 0).attr("y2", histogramHeight).attr("stroke", "black").attr("transform", `translate(${-histogramWidth / 2},0)`).attr("stroke-width", 2).attr("stroke-dasharray", "5,5");
      var mouseovertriangle = function(d2) {
        tooltipBody.style("opacity", 1);
        tooltipModal.style("opacity", 1);
        d3.select(this).style("fill", "red").style("stroke", "red");
      };
      var mouseleavetriangle = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "green").style("stroke", "green");
      };
      var mousemovetriangle = function(event2, d2) {
        tooltipModal.html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : "N/A";
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipBody.html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : "N/A";
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      var color = "green";
      var triangleSize = 25;
      var verticalTransform = histogramHeight - Math.sqrt(triangleSize) + 15;
      var triangle = d3.symbol().type(d3.symbolTriangle).size(triangleSize);
      if (isSampleExistInThisNode) {
        stLog("debug", d, "Exist");
        d3.select(this).append("path").attr("d", triangle).attr("class", "st-triangle").style("fill", color).style("stroke-width", 1).style("stroke-opacity", 1).attr("transform", function(d2) {
          return "translate(" + (-histogramWidth / 2 + xScale(treeData.show_sample[featureIndex])) + "," + verticalTransform + ")";
        }).on("mouseover", mouseovertriangle).on("mouseleave", mouseleavetriangle).on("mousemove", mousemovetriangle);
      }
    }
  }
  function processClassificationLeaf(treeData, pieWidth, pieHeight, rectWidth, rectHeight, scatterplotWidth, scatterplotHeight, colors, maxSample, showpath, d32, tooltipBody, tooltipModal, d) {
    var formatNumber = d32.format(",.0f");
    if (d.data.is_leaf) {
      const featureIndex = d.data.feature;
      if (treeData.show_sample != "nodata") {
        stLog("debug", treeData.show_sample, "treedatasample");
        isSampleExist = true;
      }
      var isSampleExistInThisNode = true;
      if (isSampleExist) {
        isSampleExistInThisNode = true;
      } else {
        isSampleExistInThisNode = false;
      }
      const classDistribution = d.data.class_distribution[0];
      const removedIndexes = [];
      const data = classDistribution.map((value, index) => {
        return {
          target_name: `${treeData.target_names[index]}`,
          classDistributionValue: value,
          index
        };
      }).filter((item) => {
        if (item.classDistributionValue === 0) {
          removedIndexes.push(item.index);
          return false;
        }
        return true;
      }).map((item) => {
        return {
          target_name: item.target_name,
          classDistributionValue: item.classDistributionValue
        };
      });
      var isSampleExist = false;
      if (treeData.show_sample != "nodata" && treeData.show_sample !== void 0) {
        stLog("debug", treeData.show_sample, "treedatasample");
        isSampleExist = true;
      }
      var isSampleExistInThisNode = true;
      if (isSampleExist) {
        isSampleExistInThisNode = true;
      } else {
        isSampleExistInThisNode = false;
      }
      let indicesArray = Array.from(
        { length: classDistribution.length },
        (_, i) => i
      );
      let removedSet = new Set(removedIndexes);
      indicesArray = indicesArray.filter(
        (value) => !removedSet.has(value)
      );
      if (isSampleExistInThisNode) {
        d.data.start_end_x_axis.forEach((currentElement, i) => {
          const sampleValue = treeData.show_sample[i];
          if (currentElement[0] != "notexist" && sampleValue > currentElement[0]) {
            isSampleExistInThisNode = false;
          }
          if (currentElement[1] != "notexist" && sampleValue < currentElement[1]) {
            isSampleExistInThisNode = false;
          }
        });
      }
      var allCurrentSamples = 0;
      for (let i = 0; i < d.data.class_distribution[0].length; i++) {
        allCurrentSamples = allCurrentSamples + parseInt(d.data.class_distribution[0][i]);
      }
      const radius = Math.max(
        20,
        Math.min(pieWidth, pieHeight) / 2 * Math.sqrt(allCurrentSamples / maxSample)
      );
      stLog("debug", allCurrentSamples, "current sample");
      stLog("debug", maxSample, "max sample");
      const pie = d32.pie().value((d2) => d2.classDistributionValue).sort(null);
      const dataPrepared = pie(data);
      const arc = d32.arc().innerRadius(0).outerRadius(radius);
      stLog("debug", "treeID", "tree id");
      stLog("debug", radius, "srednicaaaaaaaa");
      stLog("debug", allCurrentSamples, "current samples");
      stLog("debug", maxSample, "max sample");
      data.forEach(function(d2) {
        d2.classDistributionValue = +d2.classDistributionValue;
        d2.enabled = true;
      });
      const total = d32.sum(
        data.map(function(d2) {
          return d2.enabled ? d2.classDistributionValue : 0;
        })
      );
      const mouseover = function(d2) {
        tooltipBody.style("opacity", 1);
        tooltipModal.style("opacity", 1);
        d32.select(this).style("stroke", "#EF4A60");
      };
      const mouseleave = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d32.select(this).style("stroke", "#545454");
      };
      const mousemove = function(event2, d2) {
        tooltipBody.html(
          `<b>${d2.data.target_name}</b>: ${d2.data.classDistributionValue}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(
          `<b>${d2.data.target_name}</b>: ${d2.data.classDistributionValue}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      if (dataPrepared[0] && "data" in dataPrepared[0]) {
        let nodeToClick = d32.select(this).datum();
        d32.select(this).selectAll("path").data(dataPrepared).join("path").attr("class", "piechart").attr("d", arc).attr("fill", (d2, i) => colors[indicesArray[i]]).attr("transform", `translate(${10},${radius - 2})`).attr("stroke", "#545454").on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", mousemove).on("click", function() {
          showpath(nodeToClick);
        }).style("stroke-width", "0.75px").each(function(d2) {
          this._current = d2;
        });
        stLog("debug", isSampleExistInThisNode, "Exist Sample true false");
        d32.select(this).append("g").attr("class", "st-text-pie").attr("text-anchor", "middle").selectAll(".st-text-pie").data(dataPrepared).join("g").attr(
          "transform",
          (d2, i) => `translate(10,${radius * 2 + 20 + i * 40})`
        ).each(function(d2, i) {
          const group = d32.select(this);
          group.append("text").attr("class", "st-pie-target").attr("x", 0).attr("y", 0).attr("fill", "black").style("text-anchor", "middle").style("font-size", "18px").text(d2.data.target_name).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
          group.append("text").attr("class", "st-pie-target2").attr("x", 0).attr("y", 20).attr("fill", "black").style("text-anchor", "middle").style("font-size", "18px").text(formatNumber(d2.data.classDistributionValue)).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
        });
      } else {
        d32.select(this).append("rect").attr("class", "histogram-background").attr("x", -(scatterplotWidth / 2) - 5).attr("y", -10).attr("width", rectWidth - 40).attr("height", rectHeight).attr("stroke-width", 1).attr("stroke", "#545454").attr("rx", 10).attr("ry", 10).style("fill", "#ffffff");
        d32.select(this).append("text").attr("dy", ".0em").attr("y", 10).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Samples " + d2.data.samples).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
        d32.select(this).append("text").attr("dy", ".0em").attr("y", 40).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Value [" + d2.data.class_distribution + "]").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
        d32.select(this).append("text").attr("dy", ".0em").attr("y", 70).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Class " + d2.data.treeclass).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
      }
    }
  }
  function processRegressionNode(d, treeData, scatterplotWidth, scatterplotHeight, histogramWidth, histogramHeight, rectWidth, rectHeight, tooltipBody, tooltipModal, colors) {
    if (!d.data.is_leaf) {
      let calculateAverages = function(data, threshold2) {
        let sumBelowThreshold = 0;
        let countBelowThreshold = 0;
        let sumAboveOrEqualThreshold = 0;
        let countAboveOrEqualThreshold = 0;
        data.forEach((item) => {
          if (item[0] < threshold2) {
            sumBelowThreshold += item[1];
            countBelowThreshold++;
          } else {
            sumAboveOrEqualThreshold += item[1];
            countAboveOrEqualThreshold++;
          }
        });
        const averageBelowThreshold = sumBelowThreshold / countBelowThreshold;
        const averageAboveOrEqualThreshold = sumAboveOrEqualThreshold / countAboveOrEqualThreshold;
        return {
          averageBelowThreshold,
          averageAboveOrEqualThreshold
        };
      };
      const featureIndex = d.data.feature;
      let filteredData = treeData.data_feature.map(
        (row) => row[d.data.feature]
      );
      let combinedData = filteredData.map((value, index) => [
        value,
        treeData.data_target[index]
      ]);
      let indexSet = /* @__PURE__ */ new Set();
      d.data.start_end_x_axis.forEach((currentElement, index) => {
        if (currentElement[0] != "notexist") {
          treeData.data_feature.forEach((value, innerIndex) => {
            if (value[index] > currentElement[0]) {
              indexSet.add(innerIndex);
            }
          });
        }
        if (currentElement[1] != "notexist") {
          treeData.data_feature.forEach((value, innerIndex) => {
            if (value[index] < currentElement[1]) {
              indexSet.add(innerIndex);
            }
          });
        }
      });
      var isSampleExist = false;
      if (treeData.show_sample != "nodata" && treeData.show_sample != void 0) {
        stLog("debug", treeData.show_sample, "treedatasample");
        isSampleExist = true;
      }
      var isSampleExistInThisNode = true;
      if (isSampleExist) {
        isSampleExistInThisNode = true;
      } else {
        isSampleExistInThisNode = false;
      }
      if (isSampleExist) {
        d.data.start_end_x_axis.forEach((currentElement, i) => {
          const sampleValue = treeData.show_sample[i];
          if (currentElement[0] != "notexist" && sampleValue > currentElement[0]) {
            isSampleExistInThisNode = false;
          }
          if (currentElement[1] != "notexist" && sampleValue < currentElement[1]) {
            isSampleExistInThisNode = false;
          }
        });
      }
      let indexesToRemove = Array.from(/* @__PURE__ */ new Set([...indexSet]));
      indexesToRemove.sort((a, b) => b - a);
      indexesToRemove.forEach((index) => {
        if (index >= 0 && index < combinedData.length) {
          combinedData.splice(index, 1);
        }
      });
      d3.select(this).append("rect").attr("class", "histogram-background").attr("x", -(scatterplotWidth / 2) - 25).attr("y", -10).attr("width", rectWidth).attr("height", rectHeight).attr("stroke-width", 1).attr("stroke", "#545454").attr("rx", 10).attr("ry", 10).style("fill", "#ffffff");
      let combinedDataValues = combinedData.map((item) => item[0]);
      const average = calculateAverages(
        combinedData,
        d.data.threshold
      );
      xExtent = d3.extent(combinedDataValues);
      if (Number.isNaN(xExtent[0]) || xExtent[0] === void 0 || xExtent[0] > d.data.threshold) {
        xExtent[0] = d.data.threshold - 0.2;
      }
      if (Number.isNaN(xExtent[1]) || xExtent[1] === void 0 || xExtent[1] < d.data.threshold) {
        xExtent[1] = d.data.threshold + 0.2;
      }
      if (isSampleExistInThisNode) {
        if (xExtent[0] > treeData.show_sample[featureIndex]) {
          xExtent[0] = treeData.show_sample[featureIndex] - 0.2;
        }
        if (xExtent[1] < treeData.show_sample[featureIndex]) {
          xExtent[1] = treeData.show_sample[featureIndex] + 0.2;
        }
      }
      const xScale = d3.scaleLinear().domain(xExtent).nice().range([0, scatterplotWidth]);
      const xDomain = xScale.domain();
      const xTickValues = [
        xDomain[0],
        d.data.threshold,
        xDomain[1]
      ];
      d3.select(this).append("g").attr(
        "transform",
        `translate(${-scatterplotWidth / 2}, ${scatterplotHeight})`
      ).attr("class", "xAxis").call(
        d3.axisBottom(xScale).tickSize(0).tickValues(xTickValues).tickPadding(8)
      ).selectAll(".tick").attr("class", "xAxis-text").style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("fill", "black");
      d3.select(this).selectAll(".domain").style("stroke", "black");
      yExtent = d3.extent(treeData.data_target);
      const yScale = d3.scaleLinear().domain(yExtent).nice().range([scatterplotHeight, 0]);
      const yDomain = yScale.domain();
      const yTickValues = [
        yDomain[0],
        yDomain[0] + (yDomain[1] - yDomain[0]) / 2,
        yDomain[1]
      ];
      stLog("debug", yTickValues, "yTickValues");
      d3.select(this).append("g").call(
        d3.axisRight(yScale).tickSize(0).tickPadding(4).tickValues(yTickValues).tickFormat(d3.format(",.0f"))
      ).call((d2) => d2.select(".domain").remove()).attr("transform", `translate(${-scatterplotWidth / 2 - yAxisMargin}, 0)`).attr("class", "yAxis").selectAll(".tick").attr("class", "yAxis-text").style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("fill", "black");
      d3.select(this).selectAll(".domain").style("stroke", "black");
      const mouseover = function(d2) {
        tooltipBody.style("opacity", 1);
        d3.select(this).style("fill", "red");
        tooltipModal.style("opacity", 1);
        d3.select(this).style("fill", "red");
      };
      const mouseleave = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "blue");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "blue");
      };
      const mouseoverline = function(d2) {
        tooltipBody.style("opacity", 1);
        d3.select(this).style("stroke", "red");
        tooltipModal.style("opacity", 1);
        d3.select(this).style("stroke", "red");
      };
      const mouseleaveline = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("stroke", "black");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("stroke", "black");
      };
      const mousemovecircle = function(event2, d2) {
        tooltipBody.html(`<b>(X,Y):</b> (${parseFloat(d2[0]).toFixed(3)}) (${parseFloat(d2[1]).toFixed(3)})`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(`<b>(X,Y):</b> (${parseFloat(d2[0]).toFixed(3)}) (${parseFloat(d2[1]).toFixed(3)})`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      d3.select(this).append("g").selectAll("g").data(combinedData).join("circle").attr("cx", (d2, i) => xScale(combinedData[i][0])).attr("cy", (d2, i) => yScale(combinedData[i][1])).attr("r", 2).attr(
        "transform",
        `translate(${-scatterplotWidth / 2}, ${0})`
      ).style("fill", "blue").style("fill-opacity", 0.5).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", mousemovecircle);
      const mousemoveavaragebelow = function(event2, d2) {
        tooltipBody.html(`<b>Average:</b> ${parseFloat(average.averageBelowThreshold).toFixed(3)} `).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(`<b>Average:</b> ${parseFloat(average.averageBelowThreshold).toFixed(3)} `).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      const mousemoveavarageabove = function(event2, d2) {
        tooltipBody.html(
          `<b>Average:</b> ${parseFloat(average.averageAboveOrEqualThreshold).toFixed(3)} `
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(
          `<b>Averge:</b> ${parseFloat(average.averageAboveOrEqualThreshold).toFixed(3)} `
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      const mousemoveAllData = function(event2, d2) {
        tooltipBody.html(
          `<b>All Data</b>
              <br>Impurity: ${d2.data.impurity}
              <br>Samples: ${d2.data.samples}
              <br>Threshold: ${parseFloat(d2.data.threshold).toFixed(3)}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px").style("opacity", 1);
        tooltipModal.html(
          `<b>All Data</b>:<br>
              <br>Impurity: ${d2.data.impurity}
              <br>Samples: ${d2.data.samples}
              <br>Threshold: ${parseFloat(d2.data.threshold).toFixed(3)}`
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px").style("opacity", 1);
      };
      const mouseleaveAllData = function(event2, d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
      };
      var threshold = parseFloat(d.data.threshold).toFixed(3);
      d3.select(this).append("line").attr("class", "threshold-line").attr("x1", xScale(threshold)).attr("x2", xScale(threshold)).attr("y1", 0).attr("y2", scatterplotHeight).attr("stroke", "black").attr("transform", `translate(${-scatterplotWidth / 2},0)`).attr("stroke-width", 2).attr("stroke-dasharray", "5,5").style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
      stLog("debug", average.averageBelowThreshold, "avarage 1:");
      if (!isNaN(average.averageBelowThreshold)) {
        d3.select(this).append("line").attr("class", "average-line").attr("x1", 0).attr("x2", xScale(threshold)).attr("y1", yScale(average.averageBelowThreshold)).attr("y2", yScale(average.averageBelowThreshold)).attr("stroke", "black").attr("transform", `translate(${-histogramWidth / 2},0)`).attr("stroke-width", 2).attr("stroke-dasharray", "5,5").on("mouseover", mouseoverline).on("mouseleave", mouseleaveline).on("mousemove", mousemoveavaragebelow).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
      }
      if (!isNaN(average.averageAboveOrEqualThreshold)) {
        d3.select(this).append("line").attr("class", "average-line").attr("x1", xScale(threshold)).attr("x2", xScale(xDomain[1])).attr("y1", yScale(average.averageAboveOrEqualThreshold)).attr("y2", yScale(average.averageAboveOrEqualThreshold)).attr("stroke", "black").attr("transform", `translate(${-histogramWidth / 2},0)`).attr("stroke-width", 2).attr("stroke-dasharray", "5,5").on("mouseover", mouseoverline).on("mouseleave", mouseleaveline).on("mousemove", mousemoveavarageabove);
      }
      d3.select(this).append("text").attr("class", "st-target").attr("x", 0).attr("y", rectHeight + 15).style("text-anchor", "middle").style("font-size", "18px").text(treeData.feature_names[d.data.feature]).on("mousemove", mousemoveAllData).on("mouseleave", mouseleaveAllData).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
      var mouseovertriangle = function(d2) {
        tooltipBody.style("opacity", 1);
        tooltipModal.style("opacity", 1);
        d3.select(this).style("fill", "red").style("stroke", "red");
      };
      var mouseleavetriangle = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "green").style("stroke", "green");
      };
      var mousemovetriangle = function(event2, d2) {
        tooltipModal.html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : "N/A";
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipBody.html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : "N/A";
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      var color = "green";
      var triangleSize = 25;
      var verticalTransform = histogramHeight - Math.sqrt(triangleSize) + 15;
      var triangle = d3.symbol().type(d3.symbolTriangle).size(triangleSize);
      if (isSampleExistInThisNode) {
        stLog("debug", d, "Exist");
        d3.select(this).append("path").attr("d", triangle).attr("class", "st-triangle").style("stroke-width", 1).style("stroke-opacity", 1).style("fill", color).attr("transform", function(d2) {
          return "translate(" + (-histogramWidth / 2 + xScale(treeData.show_sample[featureIndex])) + "," + verticalTransform + ")";
        }).on("mouseover", mouseovertriangle).on("mouseleave", mouseleavetriangle).on("mousemove", mousemovetriangle);
      }
    }
  }
  function processRegressionLeaf(d, treeData, scatterplotLeafWidth, scatterplotLeafHeight, rectHeight, tooltipBody, tooltipModal, click, showpath) {
    if (d.data.is_leaf) {
      let calculateAverage = function(data) {
        const length = data.length;
        const sum = data.reduce((accumulator, currentValue) => {
          return accumulator + currentValue[1];
        }, 0);
        return sum / length;
      }, wrapText = function(text, width) {
        text.each(function() {
          var textElement2 = d3.select(this), words = textElement2.text().split(/\s+/).reverse(), word, line = [], lineNumber = 0, lineHeight = 1.1, x = textElement2.attr("x"), y = textElement2.attr("y"), dy = 0, tspan = textElement2.text(null).append("tspan").attr("x", x).attr("y", y);
          while (word = words.pop()) {
            line.push(word);
            tspan.text(line.join(" "));
            if (tspan.node().getComputedTextLength() > width && line.length > 1) {
              line.pop();
              tspan.text(line.join(" "));
              line = [word];
              tspan = textElement2.append("tspan").attr("class", "st-target").attr("x", x).attr("y", y).attr("dy", ++lineNumber * lineHeight + "em").text(word);
            }
          }
        });
      };
      const featureIndex = d.data.feature;
      var isSampleExist = false;
      if (treeData.show_sample != "nodata" && treeData.show_sample !== void 0) {
        isSampleExist = true;
      }
      var isSampleExistInThisNode = isSampleExist;
      let filteredData = treeData.data_feature.map(
        (row) => row[d.parent.data.feature]
      );
      let combinedData = filteredData.map((value, index) => [
        value,
        treeData.data_target[index]
      ]);
      let indexSet = /* @__PURE__ */ new Set();
      d.data.start_end_x_axis.forEach((currentElement, index) => {
        if (currentElement[0] != "notexist") {
          treeData.data_feature.forEach((value, innerIndex) => {
            if (value[index] > currentElement[0]) {
              indexSet.add(innerIndex);
            }
          });
        }
        if (currentElement[1] != "notexist") {
          treeData.data_feature.forEach((value, innerIndex) => {
            if (value[index] < currentElement[1]) {
              indexSet.add(innerIndex);
            }
          });
        }
      });
      let indexesToRemove = Array.from(/* @__PURE__ */ new Set([...indexSet]));
      indexesToRemove.sort((a, b) => b - a);
      indexesToRemove.forEach((index) => {
        if (index >= 0 && index < combinedData.length) {
          combinedData.splice(index, 1);
        }
      });
      if (isSampleExist) {
        d.data.start_end_x_axis.forEach((currentElement, i) => {
          const sampleValue = treeData.show_sample[i];
          if (currentElement[0] != "notexist" && sampleValue > currentElement[0]) {
            isSampleExistInThisNode = false;
          }
          if (currentElement[1] != "notexist" && sampleValue < currentElement[1]) {
            isSampleExistInThisNode = false;
          }
        });
      }
      var nodeToClick = d;
      d3.select(this).append("rect").attr("class", "histogram-background").attr("x", -(scatterplotLeafWidth / 2) - 10).attr("y", -10).attr("width", rectHeight + 20).attr("height", rectHeight - 10).attr("stroke-width", 1).attr("stroke", "#545454").attr("rx", 10).attr("ry", 10).style("fill", "#ffffff").on("click", click).on("click", function() {
        showpath(nodeToClick);
      });
      let combinedDataValues = combinedData.map((item) => item[0]);
      const average = calculateAverage(combinedData);
      var xExtent2 = d3.extent(combinedDataValues);
      xExtent2[0] = xExtent2[0] - 0.2;
      xExtent2[1] = xExtent2[1] + 0.2;
      if (isSampleExistInThisNode) {
        if (xExtent2[0] > treeData.show_sample[featureIndex]) {
          xExtent2[0] = treeData.show_sample[featureIndex] - 0.2;
        }
        if (xExtent2[1] < treeData.show_sample[featureIndex]) {
          xExtent2[1] = treeData.show_sample[featureIndex] + 0.2;
        }
      }
      const xScale = d3.scaleLinear().domain(xExtent2).nice().range([0, scatterplotLeafWidth]);
      yExtent = d3.extent(treeData.data_target);
      const yScale = d3.scaleLinear().domain(d3.extent(yExtent)).nice().range([scatterplotLeafHeight, 0]);
      const yDomain = yScale.domain();
      const yTickValues = [
        yDomain[0],
        yDomain[0] + (yDomain[1] - yDomain[0]) / 2,
        yDomain[1]
      ];
      stLog("debug", yTickValues, "yTickValues");
      d3.select(this).append("g").call(
        d3.axisRight(yScale).tickSize(0).tickPadding(4).tickValues(yTickValues).tickFormat(d3.format(",.0f"))
      ).call((d2) => d2.select(".domain").remove()).attr(
        "transform",
        `translate(${-scatterplotLeafWidth / 2 + 15 - yAxisMargin}, 0)`
      ).attr("class", "yAxis").style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("fill", "black");
      d3.select(this).selectAll(".domain").style("stroke", "black");
      const mouseover = function(d2) {
        tooltipBody.style("opacity", 1);
        d3.select(this).style("fill", "red");
        tooltipModal.style("opacity", 1);
        d3.select(this).style("fill", "red");
      };
      const mouseleave = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "blue");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "blue");
      };
      const mouseoverline = function(d2) {
        tooltipBody.style("opacity", 1);
        d3.select(this).style("stroke", "red");
        tooltipModal.style("opacity", 1);
        d3.select(this).style("stroke", "red");
      };
      const mouseleaveline = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("stroke", "black");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("stroke", "black");
      };
      const mousemovecircle = function(event2, d2) {
        tooltipBody.html(`<b>(X,Y):</b> (${parseFloat(d2[0]).toFixed(3)}) (${parseFloat(d2[1]).toFixed(3)})`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(`<b>(X,Y):</b> (${parseFloat(d2[0]).toFixed(3)}) (${parseFloat(d2[1]).toFixed(3)})`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      const mousemoveavarage = function(event2, d2) {
        tooltipBody.html(`<b>Average:</b> ${parseFloat(average).toFixed(3)} `).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipModal.html(`<b>Average:</b> ${parseFloat(average).toFixed(3)} `).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      d3.select(this).append("g").selectAll("g").data(combinedData).join("circle").attr("cx", (d2, i) => xScale(combinedData[i][0])).attr("cy", (d2, i) => yScale(combinedData[i][1])).attr("r", 2).attr(
        "transform",
        `translate(${-scatterplotLeafWidth / 2 + 15}, ${0})`
      ).style("fill", "blue").style("fill-opacity", 0.5).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", mousemovecircle);
      if (!isNaN(average)) {
        d3.select(this).append("line").attr("class", "average-line").attr("x1", 0).attr("x2", scatterplotLeafWidth).attr("y1", yScale(average)).attr("y2", yScale(average)).attr("stroke", "black").attr(
          "transform",
          `translate(${-scatterplotLeafWidth / 2 + 15},0)`
        ).attr("stroke-width", 2).attr("stroke-dasharray", "5,5").on("mouseover", mouseoverline).on("mouseleave", mouseleaveline).on("mousemove", mousemoveavarage);
      }
      var maxWidth = 100;
      var textElement = d3.select(this).append("text").attr("class", "st-target").attr("x", 15).attr("y", rectHeight + 5).style("text-anchor", "middle").style("font-size", "18px").text(`${d.data.treeclass} = ${parseFloat(d.data.class_distribution[0][0].toFixed(3))}`).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").call(wrapText, maxWidth);
      var lineCount = textElement.selectAll("tspan").size();
      var lineHeightEm = 1.1;
      var fontSizePx = 18;
      var textSpacing = 5;
      var nTextY = rectHeight + lineCount * lineHeightEm * fontSizePx + textSpacing;
      d3.select(this).append("text").attr("class", "st-target").attr("x", 15).attr("y", nTextY).style("text-anchor", "middle").style("font-size", "18px").text(`n = ${d.data.samples}`).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
      var mouseovertriangle = function(d2) {
        tooltipBody.style("opacity", 1);
        tooltipModal.style("opacity", 1);
        d3.select(this).style("fill", "red").style("stroke", "red");
      };
      var mouseleavetriangle = function(d2) {
        tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
        d3.select(this).style("fill", "green").style("stroke", "green");
      };
      var mousemovetriangle = function(event2, d2) {
        tooltipModal.html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : "N/A";
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
        tooltipBody.html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : "N/A";
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
      };
      var color = "green";
      var triangleSize = 25;
      var triangle = d3.symbol().type(d3.symbolTriangle).size(triangleSize);
      stLog("debug", d, "XSCALE");
      if (isSampleExistInThisNode) {
        stLog("debug", d, "Existtt");
        d3.select(this).append("path").attr("d", triangle).attr("class", "st-triangle").style("stroke", color).style("stroke-width", 1).style("stroke-opacity", 1).style("fill", color).attr("transform", function(d2) {
          return "translate(" + (-scatterplotLeafWidth / 2 + 15 + xScale(treeData.show_sample[d2.parent.data.feature])) + "," + scatterplotLeafHeight + ")";
        }).on("mouseover", mouseovertriangle).on("mouseleave", mouseleavetriangle).on("mousemove", mousemovetriangle);
      }
    }
  }

  // supertree/js/src/viewport.js
  function createViewportController(config) {
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
      onControlsLockedChange
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
        const divElement2 = document.getElementsByClassName(`st-tree-div-${myid}`)[0];
        return {
          divWidth: divElement2.clientWidth,
          divHeight: divElement2.clientHeight,
          svgElement: d3.select(divElement2).select("svg#mySVG-treeID")
        };
      }
      const divElement = document.getElementsByClassName("st-body-tree-div-treeID")[0];
      return {
        divWidth: divElement.clientWidth,
        divHeight: divElement.clientHeight,
        svgElement: d3.select(divElement).select("svg#mySVG-treeID")
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
        height: Math.max(treeBounds.height + verticalPadding * 2, 1)
      };
    }
    function getVisibleTreeBounds(options = {}) {
      const visibleNodes = getTreeRoot().descendants();
      if (options.fallbackToFullForSingleRoot && visibleNodes.length === 1 && !visibleNodes[0].parent) {
        return getFullTreeBounds();
      }
      const renderedBounds = getRenderedTreeBounds();
      if (options.useRenderedBounds !== false && renderedBounds !== null) {
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
        height: Math.max(maxY - minY, 1)
      };
    }
    function shouldAutoFitAfterInternalToggle(previousBounds) {
      var _a, _b;
      if (!previousBounds) {
        return false;
      }
      const nextBounds = getVisibleTreeBounds({
        fallbackToFullForSingleRoot: true,
        useRenderedBounds: true
      });
      const widthRatio = nextBounds.width / Math.max(previousBounds.width, 1);
      const heightRatio = nextBounds.height / Math.max(previousBounds.height, 1);
      const currentTransform = d3.zoomTransform(getTreeSVG().node());
      const fitTransform = getFitTransform("visible", {
        fallbackToFullForSingleRoot: true,
        useRenderedBounds: true
      });
      const currentScale = (_a = currentTransform == null ? void 0 : currentTransform.k) != null ? _a : 1;
      const fitScale = (_b = fitTransform == null ? void 0 : fitTransform.k) != null ? _b : 1;
      const isMaterialShrink = widthRatio < 0.72 || heightRatio < 0.72;
      const isTooZoomedInForExpandedTree = currentScale > fitScale * 1.18;
      return isMaterialShrink || isTooZoomedInForExpandedTree;
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
        height: Math.max(maxY - minY, 1)
      };
    }
    function getFitTransform(mode = "visible", options = {}) {
      var _a;
      const { divWidth, divHeight } = getViewportContext();
      const fitBounds = mode === "full" ? getFullTreeBounds() : getVisibleTreeBounds(options);
      const topViewportPadding = 20;
      const maxScale = (_a = options.maxScale) != null ? _a : 2;
      const availableWidth = Math.max(divWidth - nodeTreeMargin.left - nodeTreeMargin.right, 1);
      const availableHeight = Math.max(divHeight - nodeTreeMargin.top - nodeTreeMargin.bottom, 1);
      let currentScale = Math.min(
        availableWidth / fitBounds.width,
        availableHeight / fitBounds.height
      );
      currentScale = Math.min(Math.max(0.05, currentScale), maxScale);
      const centeredX = (fitBounds.minX + fitBounds.maxX) / 2;
      return d3.zoomIdentity.translate(
        -centeredX * currentScale + divWidth / 2,
        -fitBounds.minY * currentScale + topViewportPadding
      ).scale(currentScale);
    }
    function resetZoom(mode = "visible", transitionDuration = fallbackDuration, options = {}) {
      const { svgElement } = getViewportContext();
      const rootTransform = getFitTransform(mode, options);
      svgElement.interrupt();
      if (transitionDuration <= 0) {
        svgElement.call(zoom.transform, rootTransform);
        return rootTransform;
      }
      svgElement.transition().duration(transitionDuration).call(zoom.transform, rootTransform);
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
      svgElement.transition().duration(transitionDuration).call(zoom.transform, lastViewportTransform);
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
            useRenderedBounds: true
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
              useRenderedBounds: true
            }));
            runAfterRenderSettled(function() {
              rememberViewport(resetZoom("visible", 0, {
                fallbackToFullForSingleRoot: false,
                maxScale: 3.5,
                useRenderedBounds: true
              }));
            }, 140);
          }, 60);
          return;
        case "root-toggle":
          runAfterRender(function() {
            rememberViewport(resetZoom("full", durations.rootFit));
          });
          finishActionAfter(durations.rootFit);
          return;
        case "depth-change":
        case "sample-path":
          runAfterRender(function() {
            rememberViewport(
              resetZoom("visible", durations.fit, { fallbackToFullForSingleRoot: true })
            );
          });
          if (actionType === "depth-change" || actionType === "sample-path") {
            finishActionAfter(durations.fit);
          }
          return;
        case "fit-visible":
          runAfterRenderSettled(function() {
            rememberViewport(resetZoom("visible", Math.min(durations.fit, 220), {
              fallbackToFullForSingleRoot: true,
              useRenderedBounds: true
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
            if (shouldAutoFitAfterInternalToggle(options.previousBounds)) {
              rememberViewport(resetZoom("visible", Math.min(durations.fit, 220), {
                fallbackToFullForSingleRoot: true,
                useRenderedBounds: true
              }));
              finishActionAfter(Math.min(durations.fit, 220));
              return;
            }
            rememberViewport();
            finishActionAfter(0);
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
    modal.onclick = function(event2) {
      if (event2.target == modal) {
        modal.style.display = "none";
        applyViewportPolicy("modal-close");
      }
    };
    btn.onclick = function() {
      applyViewportPolicy("modal-open");
      modal.style.display = "block";
    };
    if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
      window.addEventListener("resize", handleViewportResize);
    }
    return {
      getIsLocked,
      setControlsLocked,
      applyViewportPolicy,
      resetZoom,
      rememberViewport,
      getVisibleTreeBounds
    };
  }

  // supertree/js/src/tree_app.js
  function buildTree(pathJson = "data/bugdata.json", pytree = "$treetemplate") {
    async function checkD3Element(selector, timeout = 5e3) {
      const interval = 100;
      const startTime = Date.now();
      const divElement = document.querySelector(selector);
      if (!divElement) {
        throw new Error(`Div with selector ${selector} does not exist.`);
      }
      while (Date.now() - startTime < timeout) {
        if (typeof d3 !== "undefined") {
          stLog("info", "D3.js successfully loaded");
          return true;
        }
        await new Promise((resolve) => setTimeout(resolve, interval));
      }
      stLog("warning", "D3.js not loaded after timeout");
      divElement.innerHTML = `
    <div style="color: red; font-weight: bold;">
      D3.js library is not loaded. Try Again :(.
    </div>`;
      return false;
    }
    checkD3Element("#graph-div-treeID").then(() => {
      d3.select("#openModalBtn").attr("class", "st-option-button");
      d3.select("#graph-div-treeID").attr("class", "st-body-tree-div-treeID");
      let myid = Math.random();
      function addModalToFirstBody() {
        let firstBody = document.getElementsByTagName("body")[0];
        let modalHtml = `
    <div id="myModal-treeID" class="st-modal">
        <div class="st-modal-content">
           <span id="closeBtn-treeID" class="st-closeBtn">&times;</span>
            <div id="st-info-div-treeID" class="st-info-div"></div>
            <div id = "toolbar-treeID" class="st-toolbar"></div>
            <div id ="graph-div-treeID" class ="st-tree-div-${myid}"></div>     
      <div id="st-side-panel-treeID" class="st-side-panel">
            <span id="st-close-button-treeID" class="st-close-button">&times;</span>
        <div>
        </div>
    </div>
    `;
        firstBody.insertAdjacentHTML("beforeend", modalHtml);
      }
      addModalToFirstBody();
      var stjupyter = false;
      if (document.body.getAttribute("data-jp-theme-name") !== null) {
        stjupyter = true;
      }
      const toolbarRoot = d3.select("#toolbar-treeID");
      const primaryToolbarGroup = toolbarRoot.append("div").attr("class", "st-toolbar-group");
      const secondaryToolbarGroup = toolbarRoot.append("div").attr("class", "st-toolbar-group");
      const depthToolbarGroup = toolbarRoot.append("div").attr("class", "st-toolbar-group");
      const tertiaryToolbarGroup = toolbarRoot.append("div").attr("class", "st-toolbar-group");
      let Modalbutton = primaryToolbarGroup.append("button").html(svgWindow).attr("class", "st-option-button").attr("id", "openModalBtn-treeID");
      if (!stjupyter) {
        Modalbutton.style("display", "none");
      }
      let modal = document.getElementById("myModal-treeID");
      let btn = document.getElementById("openModalBtn-treeID");
      let span = document.getElementById("closeBtn-treeID");
      const regr = "regression";
      const classification = "classification";
      const nodata = "nodata";
      let boldLinks = true;
      let globalX = true;
      let globalY = true;
      const yMultiplayer = 1;
      const xMultiplayer = 1;
      let isLocked = false;
      let globalMaxSample = 0;
      const pieHeight = 100;
      const pieWidth = 100;
      const histogramWidth = 150;
      const histogramHeight = 80;
      const scatterplotWidth = histogramWidth;
      const scatterplotHeight = histogramHeight;
      const scatterplotLeafWidth = histogramHeight + 10;
      const scatterplotLeafHeight = histogramHeight;
      const rectHeight = 110;
      const rectWidth = 195;
      var maxSample = 0;
      let minSample = Infinity;
      const interactionDurations = {
        toggle: 560,
        fit: 260,
        rootFit: 520,
        modal: 220
      };
      const defaultLinkStroke = "#545454";
      const allColors = [
        "#FEFEBB",
        "#c7e9b4",
        "#41b6c4",
        "#FEFECD",
        "#CFE2D4",
        "#4575B4",
        "#313695",
        "#FEE090",
        "#006400",
        "#A6BDDB",
        "#444443",
        "#FFFF00",
        "#00FF00",
        "#0000FF",
        "#FFA500",
        "#800080",
        "#FF00FF",
        "#00FFFF",
        "#FFC0CB",
        "#808080",
        "#800000",
        "#008000",
        "#000080",
        "#FFFFE0",
        "#00FA9A",
        "#ADD8E6",
        "#FF4500",
        "#EE82EE",
        "#20B2AA",
        "#778899",
        "#B22222",
        "#7FFF00",
        "#4682B4",
        "#DAA520",
        "#4B0082",
        "#D2691E",
        "#5F9EA0",
        "#FF1493",
        "#696969",
        "#DC143C",
        "#00CED1",
        "#FFD700",
        "#9932CC",
        "#8B4513",
        "#00BFFF",
        "#FF69B4",
        "#A9A9A9",
        "#B22222",
        "#32CD32",
        "#1E90FF",
        "#FF8C00",
        "#BA55D3",
        "#8B0000",
        "#48D1CC",
        "#DDA0DD",
        "#FF6347",
        "#2E8B57",
        "#6495ED",
        "#FFA07A",
        "#9370DB",
        "#8B008B",
        "#FF0000",
        "#bfd2bf",
        "#a37774",
        "#124559",
        "#00FF00",
        "#46351d",
        "#0000FF",
        "#01161e",
        "#FF00FF",
        "#FFFF00",
        "#FF0000",
        "#93b7be",
        "#785964",
        "#00FF00",
        "#598392",
        "#eaf0ce",
        "#0000FF",
        "#FF00FF",
        "#FFFF00",
        "#d75841",
        "#228B22",
        "#79625d",
        "#1E90FF",
        "#fba64c",
        "#32CD32",
        "#a46848",
        "#4682B4",
        "#453e2a",
        "#c8b496",
        "#00CED1",
        "#2e1515",
        "#be9668",
        "#541e13",
        "#715a3b",
        "#17090c",
        "#dcd1b8",
        "#3f2926",
        "#d03c32",
        "#9e8676",
        "#e3aa71",
        "#de5d70",
        "#544540",
        "#335f9e",
        "#ab9682",
        "#150a2b",
        "#171559",
        "#d6d169",
        "#450c3d",
        "#913362",
        "#2b7873",
        "#2b9e62",
        "#65aacf",
        "#1f1718",
        "#67d95f",
        "#a61e49",
        "#f2c6b8",
        "#e6e3d8",
        "#eb9494",
        "#794d81",
        "#ffd8a9",
        "#ff5b4f",
        "#4a3778",
        "#ffb366",
        "#ad82cf",
        "#7e9770",
        "#f39d91",
        "#a9548a",
        "#d38e84",
        "#8455a9",
        "#814d6e",
        "#5d7668",
        "#f2af92",
        "#533a44",
        "#ffffe1",
        "#9e2081",
        "#235a63",
        "#c92e70",
        "#c37289",
        "#a83a94",
        "#db5989",
        "#4c55ba",
        "#77c1eb",
        "#f68484",
        "#acfff1",
        "#ffce96",
        "#512b8c",
        "#1f4572",
        "#221330",
        "#422452",
        "#d4ffb0",
        "#50cc9a",
        "#74ff86",
        "#201a47",
        "#2c8a9a",
        "#000000",
        "#782b8c",
        "#fffab2",
        "#fdffef",
        "#094d18",
        "#449481",
        "#f0cc69",
        "#df5f36",
        "#555555",
        "#497a3a",
        "#beb866",
        "#819650",
        "#7294d6",
        "#5b2b7c",
        "#ffffff",
        "#aaaaaa",
        "#000000",
        "#8cdaff",
        "#f0a34a",
        "#9f1d2e",
        "#5b59b3",
        "#4b0f37",
        "#63b9bb",
        "#227944"
      ];
      const colorSize = 20;
      var tooltipModal = d3.selectAll("#myModal-treeID").append("div").attr("class", "st-tooltip").style("position", "absolute").style("opacity", 0).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("font-size", "18px").style("background-color", "rgba(0, 0, 0, 0.7)").style("color", "white").style("padding", "8px").style("border-radius", "4px").style("box-shadow", "0px 4px 8px rgba(0, 0, 0, 0.3)").style("max-width", "200px").style("text-align", "center").style("z-index", "1000").style("transition", "opacity 0.3s ease");
      var tooltipBody = d3.selectAll("body").append("div").attr("class", "st-tooltip").style("position", "absolute").style("opacity", 0).style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none").style("font-size", "18px").style("background-color", "rgba(0, 0, 0, 0.7)").style("color", "white").style("padding", "8px").style("border-radius", "4px").style("box-shadow", "0px 4px 8px rgba(0, 0, 0, 0.3)").style("max-width", "200px").style("text-align", "center").style("z-index", "1000").style("transition", "opacity 0.3s ease");
      function startWatcher() {
        const treeID = "st-body-tree-div-treeID";
        const modalID = "myModal-treeID";
        function callback(mutationsList, observer2) {
          const treeElement = d3.select("." + treeID);
          if (treeElement.empty()) {
            const modalElement = d3.select("#" + modalID);
            if (!modalElement.empty()) {
              modalElement.remove();
            }
            observer2.disconnect();
          }
        }
        const observer = new MutationObserver(callback);
        const config = { childList: true, subtree: true };
        observer.observe(document.body, config);
      }
      startWatcher();
      async function loadJSONFiles() {
        try {
          let data;
          if (pytree != "$treetemplate") {
            var treeData = pytree;
            data = {
              nodeData: treeData.node_data,
              treeData: treeData.tree_data
            };
          } else {
            const [treeDataResponse] = await Promise.all([fetch(pathJson)]);
            if (!treeDataResponse.ok) {
              throw new Error("Network response was not ok");
            }
            const treeData2 = await treeDataResponse.json();
            stLog("debug", treeData2);
            data = {
              nodeData: treeData2.node_data,
              treeData: treeData2.tree_data
            };
          }
          return data;
        } catch (error) {
          stLog("error", "Error loading JSON files:");
        }
      }
      loadJSONFiles().then((data) => {
        if (data) {
          let zoomed = function(event2) {
            treeSVG.attr("transform", event2.transform);
          }, getNodeDisplayLabel = function(node) {
            if (!node) {
              return "Root";
            }
            if (!node.parent) {
              return "Root";
            }
            if (node.data && node.data.is_leaf) {
              return `Leaf ${node.id}`;
            }
            if (node.data && Number.isInteger(node.data.feature) && treeData.feature_names && treeData.feature_names[node.data.feature]) {
              return treeData.feature_names[node.data.feature];
            }
            return `Node ${node.id}`;
          }, getNodeById = function(nodeId) {
            return treeRoot.descendants().find((node) => node.id === nodeId) || null;
          }, getAncestorIds = function(node) {
            const ids = /* @__PURE__ */ new Set();
            let current = node;
            while (current) {
              ids.add(current.id);
              current = current.parent;
            }
            return ids;
          }, getDescendantIds = function(node) {
            const ids = /* @__PURE__ */ new Set();
            (function walk(current) {
              ids.add(current.id);
              if (current.children) {
                current.children.forEach(walk);
              }
              if (current._children) {
                current._children.forEach(walk);
              }
            })(node);
            return ids;
          }, wait = function(ms) {
            return new Promise((resolve) => setTimeout(resolve, ms));
          }, getRelativeDepthLayers = function(nodes) {
            const layers = [];
            const queue = nodes.map((node) => ({ node, depth: 1 }));
            while (queue.length > 0) {
              const { node, depth } = queue.shift();
              if (!layers[depth - 1]) {
                layers[depth - 1] = [];
              }
              layers[depth - 1].push(node);
              if (node.children) {
                node.children.forEach((child) => {
                  queue.push({ node: child, depth: depth + 1 });
                });
              }
            }
            return layers;
          }, selectTreeNodesByIds = function(ids) {
            return treeSVG.selectAll(".treeNode").filter((node) => ids.has(node.id));
          }, selectTreeLinksByIds = function(ids) {
            return treeSVG.selectAll("#st-link-treeID").filter((node) => ids.has(node.id));
          }, getNodeTransform = function(node, scale = 1) {
            return "translate(" + node.x * xMultiplayer + "," + node.y * yMultiplayer + ") scale(" + scale + ")";
          }, runTransition = function(selection, configureTransition) {
            if (selection.size() === 0) {
              return Promise.resolve();
            }
            const transition = configureTransition(selection.transition());
            return transition.end().catch(() => {
            });
          }, renderFocusState = function() {
            const focusedNode = currentFocusNodeId === null ? null : getNodeById(currentFocusNodeId);
            const activeIds = focusedNode ? getAncestorIds(focusedNode) : /* @__PURE__ */ new Set();
            const contextIds = focusedNode ? getDescendantIds(focusedNode) : /* @__PURE__ */ new Set();
            const anyFocus = focusedNode !== null;
            treeSVG.selectAll(".treeNode").classed("st-node-active", (d) => anyFocus && d.id === focusedNode.id).classed("st-node-context", (d) => anyFocus && contextIds.has(d.id) && d.id !== focusedNode.id).classed("st-node-path", (d) => anyFocus && activeIds.has(d.id) && d.id !== focusedNode.id).classed("st-node-dim", (d) => anyFocus && !contextIds.has(d.id) && !activeIds.has(d.id));
            treeSVG.selectAll("#st-link-treeID").classed("st-link-active", (d) => anyFocus && d.id === focusedNode.id).classed("st-link-context", (d) => anyFocus && contextIds.has(d.id) && d.id !== focusedNode.id).classed("st-link-path", (d) => anyFocus && activeIds.has(d.id) && d.id !== focusedNode.id).classed("st-link-dim", (d) => anyFocus && !contextIds.has(d.id) && !activeIds.has(d.id));
            d3.select("#st-nav-action-treeID").text(currentFocusAction);
            d3.select("#st-nav-focus-treeID").text(currentFocusLabel);
          }, setInteractionFocus = function(node, actionLabel) {
            currentFocusNodeId = node ? node.id : null;
            currentFocusAction = actionLabel;
            currentFocusLabel = getNodeDisplayLabel(node);
            renderFocusState();
          }, update = function(source2, resetTree, transitionDuration = duration, options = {}) {
            const hiddenNodeIds = options.hiddenNodeIds || /* @__PURE__ */ new Set();
            const hiddenLinkIds = options.hiddenLinkIds || hiddenNodeIds;
            var maxDepthReset = 0;
            if (resetTree) {
              let DFS = function(node, depth, maxDepth2) {
                depth++;
                if (node.children) {
                  node.children.forEach(function(child) {
                    maxDepthReset = Math.max(depth, maxDepthReset);
                    DFS(child, depth, maxDepth2);
                  });
                }
              };
              DFS(treeRoot, 0, maxDepthReset);
              treeSVG.selectAll(".treeNode").remove();
              treeSVG.selectAll("#st-link-treeID").remove();
              treeSVG.selectAll("g").remove();
              treeRoot = createHierarchyFromData();
              collapse(treeRoot, 0, maxDepthReset + 1);
              applyStableLayout(treeRoot, stableLayout);
              stLog("debug", treeSVG, "treeSVG");
            }
            d3.selectAll("#st-link-treeID").style("stroke", defaultLinkStroke);
            minSample = Infinity;
            applyStableLayout(treeRoot, stableLayout);
            let links = treeRoot.descendants().slice(1);
            var treeNode = treeSVG.selectAll(".treeNode").data(treeRoot.descendants(), function(d) {
              return d.id;
            });
            function getDescendants(sourceRoot, myDescendants = []) {
              if (sourceRoot._children) {
                sourceRoot._children.forEach(function(d) {
                  myDescendants.push(d);
                  getDescendants(d, myDescendants);
                });
              }
              if (sourceRoot.children) {
                sourceRoot.children.forEach(function(d) {
                  myDescendants.push(d);
                  getDescendants(d, myDescendants);
                });
              }
              return myDescendants;
            }
            let enterDescendants = getDescendants(source2);
            let idsArrayDescendants = enterDescendants.map(function(d) {
              return d.id;
            });
            idsArrayDescendants.push(source2.id);
            const enteredLinkIds = new Set(
              idsArrayDescendants.filter((id) => id !== source2.id)
            );
            const enteringNodeScale = 0.96;
            const enteredNodeRevealDuration = Math.max(
              Math.min(transitionDuration * 0.55, 220),
              0
            );
            var treeNodeEnter = treeNode.enter().append("g").attr("class", "treeNode").style("opacity", function(d) {
              return hiddenNodeIds.has(d.id) ? 0 : 0;
            }).attr(
              "transform",
              function(d) {
                if (hiddenNodeIds.has(d.id)) {
                  return getNodeTransform(d, enteringNodeScale);
                }
                return "translate(" + source2.cx + "," + source2.cy + ") scale(" + enteringNodeScale + ")";
              }
            ).on("click", click);
            globalXExtent = Array.from({ length: featureNumber }, () => [
              Infinity,
              -Infinity
            ]);
            globalYExtent = Array.from({ length: featureNumber }, () => [
              0,
              -Infinity
            ]);
            for (let i = 0; i < featureNumber; i++) {
              let tempArr = [];
              for (let j = 0; j < treeData.data_feature.length; j++) {
                tempArr.push(treeData.data_feature[j][i]);
              }
              let tempGlobalXExtent = d3.extent(tempArr);
              globalXExtent[i][0] = Math.min(
                globalXExtent[i][0],
                tempGlobalXExtent[0]
              );
              globalXExtent[i][1] = Math.max(
                globalXExtent[i][1],
                tempGlobalXExtent[1]
              );
            }
            for (let i = 0; i < featureNumber; i++) {
              globalXExtent[i][0] -= 0.2;
              globalXExtent[i][1] += 0.2;
            }
            if (treeData.tree_type == regr) {
              treeNodeEnter.each(function(d) {
                processRegressionNode.call(
                  this,
                  d,
                  treeData,
                  scatterplotWidth,
                  scatterplotHeight,
                  histogramWidth,
                  histogramHeight,
                  rectWidth,
                  rectHeight,
                  tooltipBody,
                  tooltipModal,
                  colors
                );
              });
              treeNodeEnter.each(function(d) {
                processRegressionLeaf.call(
                  this,
                  d,
                  treeData,
                  scatterplotLeafWidth,
                  scatterplotLeafHeight,
                  rectHeight,
                  tooltipBody,
                  tooltipModal,
                  click,
                  showpath
                );
              });
            }
            if (treeData.tree_type == classification) {
              treeNodeEnter.each(function(d) {
                processClassificationNode.call(
                  this,
                  treeData,
                  tooltipBody,
                  tooltipModal,
                  globalX,
                  globalXExtent,
                  globalY,
                  globalYExtent,
                  click,
                  histogramWidth,
                  histogramHeight,
                  rectWidth,
                  rectHeight,
                  colors,
                  d
                );
              });
              treeNodeEnter.each(function(d) {
                if (d.data.is_leaf) {
                  var sum = 0;
                  for (let i = 0; i < d.data.class_distribution[0].length; i++) {
                    sum = sum + parseFloat(d.data.class_distribution[0][i]);
                  }
                  maxSample = Math.max(maxSample, sum);
                  minSample = Math.min(minSample, sum);
                }
              });
              if (maxSample > 0) {
                globalMaxSample = Math.max(maxSample, globalMaxSample);
              }
              treeNodeEnter.each(function(d) {
                processClassificationLeaf.call(
                  this,
                  treeData,
                  pieWidth,
                  pieHeight,
                  rectWidth,
                  rectHeight,
                  scatterplotWidth,
                  scatterplotHeight,
                  colors,
                  maxSample,
                  showpath,
                  d3,
                  tooltipBody,
                  tooltipModal,
                  d
                );
              });
            }
            if (treeData.tree_type.startsWith(nodata)) {
              treeNodeEnter.each(function(d) {
                if (!d.data.is_leaf) {
                  d3.select(this).append("rect").attr("class", "histogram-background").attr("x", -(scatterplotWidth / 2) - 25).attr("y", -10).attr("width", rectWidth).attr("height", rectHeight).attr("stroke-width", 1).attr("stroke", "#545454").attr("rx", 10).attr("ry", 10).style("fill", "#ffffff");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 10).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Threshold " + parseFloat(d2.data.threshold).toFixed(3)).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 30).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Impurity: " + d2.data.impurity).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 50).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Samples " + d2.data.samples).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 70).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Value [" + d2.data.class_distribution + "]").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 90).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Class " + d2.data.treeclass).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                }
                if (d.data.is_leaf) {
                  d3.select(this).append("rect").attr("class", "histogram-background").attr("x", -(scatterplotWidth / 2) - 5).attr("y", -10).attr("width", rectWidth - 40).attr("height", rectHeight).attr("stroke-width", 1).attr("stroke", "#545454").attr("rx", 10).attr("ry", 10).style("fill", "#ffffff");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 10).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Samples " + d2.data.samples).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 40).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Value [" + d2.data.class_distribution + "]").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                  d3.select(this).append("text").attr("dy", ".0em").attr("y", 70).attr("class", "st-target").style("text-anchor", "middle").text((d2) => "Class " + d2.data.treeclass).style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
                }
              });
            }
            var treeNodeUpdate = treeNodeEnter.merge(treeNode);
            treeNodeUpdate.transition().duration(function(d) {
              if (hiddenNodeIds.has(d.id)) {
                return 0;
              }
              if (enteredLinkIds.has(d.id)) {
                return enteredNodeRevealDuration;
              }
              return transitionDuration;
            }).ease(d3.easeCubicInOut).style("opacity", function(d) {
              return hiddenNodeIds.has(d.id) ? 0 : 1;
            }).tween("logging", function(d) {
              let interpolateX = d3.interpolate(
                source2.x0 * xMultiplayer,
                d.x * xMultiplayer
              );
              let interpolateY = d3.interpolate(
                source2.y * yMultiplayer,
                d.y * yMultiplayer
              );
              return function(t) {
                let x = interpolateX(t);
                let y = interpolateY(t);
                d.cx = x;
                d.cy = y;
              };
            }).attr("transform", function(d) {
              if (hiddenNodeIds.has(d.id)) {
                return getNodeTransform(d, enteringNodeScale);
              }
              return "translate(" + d.x * xMultiplayer + "," + d.y * yMultiplayer + ") scale(1)";
            });
            let exitDescendants = [];
            exitDescendants = getDescendants(source2);
            let descendantIds = new Set(exitDescendants.map((d) => d.id));
            let treeNodeExit = treeNode.exit().filter(function(d) {
              return descendantIds.has(d.id);
            }).transition().duration(transitionDuration).ease(d3.easeCubicInOut).attr("transform", function(d) {
              return "translate(" + (source2.x + rectWidth / 2) + "," + (source2.y0 + rectHeight / 2) + ")";
            }).attr("font-size", "1em").remove();
            treeNodeExit.selectAll(".st-target").style("fill-opacity", 1e-6).style("font-size", "0px").attr("x", -rectWidth / 2).attr("y", 0);
            treeNodeExit.selectAll(".st-triangle").style("stroke-width", 0).style("fill-opacity", 0).attr("transform", function(d) {
              return "translate(" + -source2.x / 6 + "," + -source2.y0 / 6 + ")";
            }).size(0);
            const smallScaleX = d3.scaleLinear().domain(0, 10).range([0, 1]);
            const smallScaleY = d3.scaleLinear().domain(10, 0).range([1, 0]);
            treeNodeExit.selectAll(".xAxis").attr("transform", "translate(" + -rectWidth / 2 + ",0)").call(
              d3.axisBottom(smallScaleX).tickSize(0).tickPadding(8).ticks(2).tickFormat(d3.format(",.1f"))
            );
            treeNodeExit.selectAll(".yAxis").attr("transform", "translate(" + -rectWidth / 2 + ",0)").call(
              d3.axisRight(smallScaleY).tickSize(0).tickPadding(8).ticks(2).tickFormat(d3.format(",.1f"))
            );
            treeNodeExit.selectAll("rect.histogram-background").attr("width", 1e-6).attr("height", 1e-6).attr("x", -rectWidth / 2).attr("y", 0);
            treeNodeExit.selectAll("rect.bar").attr("width", 1e-6).attr("height", 1e-6).attr("x", -histogramWidth / 2 + 70).attr("y", 0);
            treeNodeExit.selectAll("circle").attr("r", 1e-6).attr("cx", -scatterplotWidth / 2 + 20).attr("cy", 0);
            treeNodeExit.selectAll("path.piechart").attr("transform", "translate(" + -rectWidth / 2 + ",0)").attr("d", d3.arc().innerRadius(0).outerRadius(1));
            let gridAnimation = treeNodeExit.select(".st-grid");
            treeNodeExit.selectAll("line.threshold-line").attr("transform", "translate(" + -rectWidth + ",-40)").attr("x1", rectWidth / 2).attr("x2", rectWidth / 2).attr("y1", rectHeight / 2 - 15).attr("y2", rectHeight / 2).attr("stroke-width", 0);
            treeNodeExit.selectAll("line.average-line").attr("transform", "translate(" + -rectWidth + ",-40)").attr("x1", rectWidth / 2).attr("x2", rectWidth / 2).attr("y1", rectHeight / 2 - 15).attr("y2", rectHeight / 2 - 15).attr("stroke-width", 0);
            gridAnimation.selectAll("line").attr("x2", 0).attr("y2", 0);
            treeNodeExit.selectAll(".yAxis-text").attr("transform", "translate(0,-40)").style("fill-opacity", 1e-6).style("font-size", "0px").attr("x", 0);
            treeNodeExit.selectAll(".xAxis-text").attr("transform", "translate(0,-40)").style("fill-opacity", 1e-6).style("font-size", "0px");
            treeNodeExit.selectAll(".st-pie-target").attr("y", function(d, i) {
              const radius = Math.max(
                20,
                Math.min(pieWidth, pieHeight) / 2 * Math.sqrt(d.data.classDistributionValue / globalMaxSample)
              );
              return -10 - radius * 2 - i * 40;
            }).attr("transform", "translate(" + -rectWidth / 2 + ",0)").style("fill-opacity", 1e-6).style("font-size", "0px");
            treeNodeExit.selectAll(".st-pie-target2").attr("y", function(d, i) {
              const radius = Math.max(
                20,
                Math.min(pieWidth, pieHeight) / 2 * Math.sqrt(d.data.classDistributionValue / globalMaxSample)
              );
              return -10 - radius * 2 - i * 40;
            }).attr("transform", "translate(" + -rectWidth / 2 + ",0)").style("fill-opacity", 1e-6).style("font-size", "0px");
            let link = treeSVG.selectAll("#st-link-treeID").data(links, function(d) {
              return d.id;
            });
            const mouseover2 = function(d) {
              const currentStroke = d3.select(this).style("stroke");
              this.dataset.previousStroke = currentStroke;
              tooltipBody.style("opacity", 1);
              tooltipModal.style("opacity", 1);
              d3.select(this).style("stroke", "#0099cc");
            };
            const mouseleave2 = function(d) {
              const previousStroke = this.dataset.previousStroke || defaultLinkStroke;
              tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
              tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
              d3.select(this).style("stroke", previousStroke);
              delete this.dataset.previousStroke;
            };
            const mousemove = function(event2, d) {
              if (treeData.tree_type == classification) {
                var currentDistribution = 0;
                for (let i = 0; i < d.data.class_distribution[0].length; i++) {
                  currentDistribution = currentDistribution + parseInt(d.data.class_distribution[0][i]);
                }
                tooltipModal.html(
                  `<b>Samples in link:</b> ${currentDistribution}`
                ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
                tooltipBody.html(
                  `<b>Samples in link:</b> ${currentDistribution}`
                ).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
              }
              if (treeData.tree_type == regr) {
                var currentDistribution = d.data.samples;
                tooltipModal.html(`<b>Samples in link:</b> ${currentDistribution}`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
                tooltipBody.html(`<b>Samples in link:</b> ${currentDistribution}`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
              }
            };
            var linkEnter = link.enter().insert("path", "g").attr("class", "st-link").attr("id", "st-link-treeID").attr("d", function(d) {
              if (hiddenLinkIds.has(d.id)) {
                return diagonal(getSourceAnchor(d.parent, d), getTargetAnchor(d));
              }
              var o = getSourceAnchor(source2, d);
              return diagonal(o, o);
            }).style("fill", "none").style("stroke", defaultLinkStroke).style("stroke-width", "2px").style("stroke-opacity", function(d) {
              return hiddenLinkIds.has(d.id) ? 0 : 0;
            }).on("mouseover", mouseover2).on("mouseleave", mouseleave2).on("mousemove", mousemove);
            var allSamples = 0;
            var allSamplesRegr = 0;
            if (treeData.tree_type == regr) {
              allSamplesRegr = nodeData.samples;
            }
            for (let i = 0; i < nodeData.class_distribution[0].length; i++) {
              allSamples = allSamples + parseInt(nodeData.class_distribution[0][i]);
            }
            if (boldLinks == true) {
              if (treeData.tree_type == classification) {
                linkEnter.each(function(d) {
                  var currentDistribution = 0;
                  for (let i = 0; i < d.data.class_distribution[0].length; i++) {
                    currentDistribution = currentDistribution + parseInt(d.data.class_distribution[0][i]);
                  }
                  d3.select(this).style(
                    "stroke-width",
                    Math.max(20 * (currentDistribution / allSamples), 1)
                  );
                });
              }
              if (treeData.tree_type == regr) {
                linkEnter.each(function(d) {
                  var currentDistribution = 0;
                  const currentSamples = d.data.samples;
                  d3.select(this).style(
                    "stroke-width",
                    Math.max(20 * (currentSamples / allSamplesRegr), 1)
                  );
                });
              }
            }
            var linkUpdate = linkEnter.merge(link);
            const linkEnterDelay = enteredNodeRevealDuration;
            linkUpdate.transition().delay(function(d) {
              if (hiddenLinkIds.has(d.id)) {
                return 0;
              }
              return enteredLinkIds.has(d.id) ? linkEnterDelay : 0;
            }).duration(function(d) {
              if (hiddenLinkIds.has(d.id)) {
                return 0;
              }
              if (!enteredLinkIds.has(d.id)) {
                return transitionDuration;
              }
              return Math.max(transitionDuration - linkEnterDelay, 0);
            }).ease(d3.easeCubicInOut).style("stroke-opacity", function(d) {
              return hiddenLinkIds.has(d.id) ? 0 : 1;
            }).attrTween("d", function(d) {
              if (hiddenLinkIds.has(d.id)) {
                const hiddenPath = diagonal(
                  getSourceAnchor(d.parent, d),
                  getTargetAnchor(d)
                );
                return function() {
                  return hiddenPath;
                };
              }
              if (enteredLinkIds.has(d.id)) {
                const settledPath = diagonal(
                  getSourceAnchor(d.parent, d),
                  getTargetAnchor(d)
                );
                return function() {
                  return settledPath;
                };
              }
              return function() {
                return diagonal(
                  getSourceAnchor(d.parent, d, { useCurrentPosition: true }),
                  getTargetAnchor(d, { useCurrentPosition: true })
                );
              };
            });
            var linkExit = link.exit().filter(function(d) {
              return descendantIds.has(d.id);
            }).transition().duration(transitionDuration).ease(d3.easeCubicInOut).style("stroke-opacity", 0).attr("d", function(d) {
              var o = getSourceAnchor(source2, d, { useExitPosition: true });
              return diagonal(o, o);
            }).remove();
            function getNodeBasePosition(node, options2 = {}) {
              if (options2.useCurrentPosition) {
                return {
                  x: node.cx,
                  y: node.cy
                };
              }
              if (options2.useExitPosition) {
                return {
                  x: node.x * xMultiplayer,
                  y: node.y0 * yMultiplayer
                };
              }
              return {
                x: node.x * xMultiplayer,
                y: node.y * yMultiplayer
              };
            }
            function getSourceAnchor(node, childNode = null, options2 = {}) {
              const base = getNodeBasePosition(node, options2);
              const cornerInset = 16;
              let anchorX = base.x;
              if (childNode) {
                const childBase = getNodeBasePosition(childNode, options2);
                if (childBase.x < base.x) {
                  anchorX = base.x - rectWidth / 2 + cornerInset;
                } else if (childBase.x > base.x) {
                  anchorX = base.x + rectWidth / 2 - cornerInset;
                }
              }
              return {
                x: anchorX,
                y: base.y + rectHeight - 10
              };
            }
            function getTargetAnchor(node, options2 = {}) {
              const base = getNodeBasePosition(node, options2);
              if (node.data && node.data.is_leaf) {
                if (treeData.tree_type == classification) {
                  return {
                    x: base.x + 10,
                    y: base.y - 3
                  };
                }
                return {
                  x: base.x + 15,
                  y: base.y + 4
                };
              }
              return {
                x: base.x,
                y: base.y - 10
              };
            }
            function diagonal(s, d) {
              const verticalDistance = Math.max(d.y - s.y, 0);
              const splitOffset = Math.min(Math.max(verticalDistance * 0.28, 34), 64);
              const joinOffset = Math.min(Math.max(verticalDistance * 0.38, 40), 82);
              const path = `M ${s.x} ${s.y}
            C ${s.x} ${s.y + splitOffset},
              ${d.x} ${d.y - joinOffset},
              ${d.x} ${d.y}`;
              return path;
            }
          }, showSample = function() {
            var sampleNode = null;
            showSampleDFS(treeRoot);
            function showSampleDFS(node) {
              if (node.children) {
                node.children.forEach(function(child) {
                  showSampleDFS(child);
                });
              } else if (node._children) {
                node._children.forEach(function(child) {
                  showSampleDFS(child);
                });
              } else {
                var isSampleExistInThisNode = true;
                node.data.start_end_x_axis.forEach((currentElement, i) => {
                  const sampleValue = treeData.show_sample[i];
                  if (currentElement[0] != "notexist" && sampleValue > currentElement[0]) {
                    isSampleExistInThisNode = false;
                  }
                  if (currentElement[1] != "notexist" && sampleValue < currentElement[1]) {
                    isSampleExistInThisNode = false;
                  }
                });
                if (isSampleExistInThisNode) {
                  sampleNode = node;
                }
              }
            }
            stLog("debug", sampleNode, "SampleNode");
            if (isLocked) {
              return;
            }
            ;
            setControlsLocked(true);
            currentDepthValue = sampleNode.depth + 1;
            showDepth(treeRoot, 0, currentDepthValue, true);
            update(treeRoot, false, 0);
            syncDepthControls();
            showpath(sampleNode);
            applyViewportPolicy("sample-path");
          }, showpath = function(d) {
            d3.selectAll("#st-link-treeID").style("stroke", defaultLinkStroke);
            const pathData = getLinksIds(d);
            const ids = pathData.ids;
            const nodedata = pathData.nodedata;
            var newHeight = nodedata.length * 250;
            newHeight = Math.max(newHeight, 1e3);
            sideSVG.attr("height", newHeight);
            d3.selectAll("#st-link-treeID").filter(function(d2) {
              return ids.includes(d2.id);
            }).style("stroke", "#0099cc");
            const sidePanel = d3.selectAll("#st-side-panel-treeID");
            if (sidePanel.classed("show")) {
              sidePanel.classed("show", false).classed("hide", true);
              setTimeout(function() {
                sideSVG.selectAll("g").remove();
                sidePanel.classed("hide", false).classed("show", true);
                renderNodeData(nodedata);
              }, 300);
            } else {
              sidePanel.classed("hide", false).classed("show", true);
              renderNodeData(nodedata);
            }
          }, renderNodeData = function(nodedata) {
            nodedata.slice().reverse().forEach(function(node, i) {
              let translateX = 150;
              let translateY = 200 * i + 100;
              if (treeData.tree_type == regr && node.data.is_leaf) {
                translateX -= 10;
              }
              const group = sideSVG.append("g").attr("class", "node-group").attr("transform", `translate(${translateX}, ${translateY})`).datum(node);
              group.each(function(d) {
                if (treeData.tree_type == classification) {
                  if (!node.data.is_leaf) {
                    processClassificationNode.call(
                      this,
                      treeData,
                      tooltipBody,
                      tooltipModal,
                      globalX,
                      globalXExtent,
                      globalY,
                      globalYExtent,
                      click,
                      histogramWidth,
                      histogramHeight,
                      rectWidth,
                      rectHeight,
                      colors,
                      node
                    );
                  }
                  if (node.data.is_leaf) {
                    processClassificationLeaf.call(
                      this,
                      treeData,
                      pieWidth,
                      pieHeight,
                      rectWidth,
                      rectHeight,
                      scatterplotWidth,
                      scatterplotHeight,
                      colors,
                      globalMaxSample,
                      showpath,
                      d3,
                      tooltipBody,
                      tooltipModal,
                      d
                    );
                  }
                }
                if (treeData.tree_type == regr) {
                  if (!node.data.is_leaf) {
                    processRegressionNode.call(
                      this,
                      d,
                      treeData,
                      scatterplotWidth,
                      scatterplotHeight,
                      histogramWidth,
                      histogramHeight,
                      rectWidth,
                      rectHeight,
                      tooltipBody,
                      tooltipModal,
                      colors
                    );
                  }
                  if (node.data.is_leaf) {
                    processRegressionLeaf.call(
                      this,
                      d,
                      treeData,
                      scatterplotLeafWidth,
                      scatterplotLeafHeight,
                      rectHeight,
                      tooltipBody,
                      tooltipModal,
                      click,
                      showpath
                    );
                  }
                }
              });
            });
          }, getLinksIds = function(node) {
            var ids = [];
            var nodedata = [];
            function linkDFS(node2) {
              stLog("debug", node2, "Node w dfsie");
              ids.push(node2.id);
              nodedata.push(node2);
              if (node2.parent) {
                linkDFS(node2.parent);
              }
            }
            linkDFS(node);
            let pathData = {
              ids,
              nodedata
            };
            return pathData;
          }, boldClick = function() {
            boldLinks = !boldLinks;
            update(treeRoot, true);
          }, saveSvg = function() {
            var svgElement = document.getElementById("mySVG-treeID");
            var serializer = new XMLSerializer();
            var svgString = serializer.serializeToString(svgElement);
            var blob = new Blob([svgString], {
              type: "image/svg+xml;charset=utf-8"
            });
            var downloadLink = document.createElement("a");
            downloadLink.href = URL.createObjectURL(blob);
            downloadLink.download = "myDiagram.svg";
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
          }, xClick = function() {
            globalX = !globalX;
            update(treeRoot, true);
          }, yClick = function() {
            globalY = !globalY;
            update(treeRoot, true);
          }, syncDepthControls = function() {
            depthLabel.text(`Depth=${Math.max(currentDepthValue - 1, 0)}`);
            depthDecreaseButton.attr(
              "disabled",
              isLocked || currentDepthValue <= 1 ? "disabled" : null
            );
            depthIncreaseButton.attr(
              "disabled",
              isLocked || currentDepthValue >= maxDepth ? "disabled" : null
            );
          }, applyDepthChange = function(depth) {
            if (depth == null || isLocked) {
              return;
            }
            const nextDepth = Math.max(1, Math.min(depth, maxDepth));
            if (nextDepth === currentDepthValue) {
              syncDepthControls();
              return;
            }
            setControlsLocked(true);
            currentDepthValue = nextDepth;
            syncDepthControls();
            showDepth(treeRoot, 0, nextDepth, true);
            update(treeRoot, false, 0);
            applyViewportPolicy("depth-change");
          }, handleDepthChange = function(event2, optDepth = "optional") {
            if (optDepth !== "optional") {
              applyDepthChange(optDepth);
              return;
            }
            const depth = extractNumber(this.value);
            if (depth != null) {
              applyDepthChange(depth + 1);
            }
          }, extractNumber = function(str) {
            const match = str.match(/\d+/);
            if (match) {
              return parseInt(match[0], 10);
            }
            return null;
          };
          const { nodeData, treeData } = data;
          const treeLeafSpacing = 300;
          const treeLevelSpacing = 235;
          const treeDepthSpacing = 400;
          const {
            treeDataConverted,
            createHierarchyFromData,
            computeStableLayout,
            applyStableLayout,
            showDepth,
            collapse,
            getTreeDepth
          } = createTreeLayoutHelpers({
            treeDataRoot: nodeData,
            treeLeafSpacing,
            treeDepthSpacing,
            treeLevelSpacing,
            xMultiplayer,
            yMultiplayer
          });
          const stableLayout = computeStableLayout(treeDataConverted);
          let featureNumber = treeData.data_feature[0].length;
          var globalXExtent = Array.from({ length: featureNumber }, () => [
            Infinity,
            -Infinity
          ]);
          var globalYExtent = Array.from({ length: featureNumber }, () => [
            0,
            -Infinity
          ]);
          async function click(event2, d) {
            if (isLocked) return;
            if (d.children == null && d._children == null) {
              return;
            }
            stLog("debug", " Collapse click event");
            setControlsLocked(true);
            const previousBounds = d.parent ? getVisibleTreeBounds({
              fallbackToFullForSingleRoot: true,
              useRenderedBounds: false
            }) : null;
            try {
              await runStagedToggle(d, previousBounds);
            } catch (error) {
              stLog("error", error, "Toggle animation failed");
              setControlsLocked(false);
              throw error;
            }
          }
          let nodeTreeMargin = { top: 20, right: 90, bottom: 160, left: 90 }, treeWidth = stableLayout.layoutWidth, treeHeight = stableLayout.layoutHeight;
          const duration = interactionDurations.toggle;
          const stagedToggleDurations = {
            expandLink: 110,
            expandNode: 140,
            collapseNode: 120,
            collapseLink: 90,
            gap: 14
          };
          let currentFocusNodeId = null;
          let currentFocusAction = "Ready";
          let currentFocusLabel = "Root";
          const paletteIndex = Math.max((treeData.palette || 1) - 1, 0);
          let colors = allColors.slice(paletteIndex * colorSize, (paletteIndex + 1) * colorSize);
          if (treeData.feature_names.length > colors.length) {
            let getRandomColor = function() {
              let letters = "0123456789ABCDEF";
              let color = "#";
              for (let i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
              }
              return color;
            };
            let missingColors = treeData.feature_names.length - colors.length;
            for (let i = 0; i < missingColors; i++) {
              colors.push(getRandomColor());
            }
          }
          var zoom = d3.zoom().scaleExtent([0.05, 5]).on("zoom", zoomed);
          var treeRoot = createHierarchyFromData();
          applyStableLayout(treeRoot, stableLayout);
          treeRoot.x0 = treeRoot.x;
          treeRoot.y0 = treeRoot.y;
          var treeSVG = d3.selectAll("#graph-div-treeID").append("svg").attr(
            "width",
            treeWidth + nodeTreeMargin.right + nodeTreeMargin.left
          ).attr(
            "height",
            treeHeight + nodeTreeMargin.top + nodeTreeMargin.bottom
          ).attr("id", "mySVG-treeID").attr("class", "st-svg").call(zoom).append("g").attr(
            "transform",
            "translate(" + nodeTreeMargin.left + "," + nodeTreeMargin.top + ")"
          );
          var sideSVG = d3.selectAll("#st-side-panel-treeID").append("svg").attr("id", "st-side-svg-treeID").attr("class", "st-svg-2").attr("width", "100%").attr("height", 1e3).style("background-color", "#f3f9fb");
          const viewportController = createViewportController({
            modal,
            btn,
            span,
            myid,
            rectWidth,
            rectHeight,
            nodeTreeMargin,
            durations: interactionDurations,
            stableLayout,
            zoom,
            getTreeRoot: () => treeRoot,
            getTreeSVG: () => treeSVG,
            onControlsLockedChange: (locked) => {
              isLocked = locked;
              if (typeof syncDepthControls === "function") {
                syncDepthControls();
              }
            }
          });
          const setControlsLocked = function(locked) {
            viewportController.setControlsLocked(locked);
            d3.selectAll("#st-depth-decrease-treeID").attr("disabled", locked ? "disabled" : null);
            d3.selectAll("#st-depth-increase-treeID").attr("disabled", locked ? "disabled" : null);
          };
          const applyViewportPolicy = viewportController.applyViewportPolicy;
          const getVisibleTreeBounds = viewportController.getVisibleTreeBounds;
          async function animateExpandLayers(sourceNode, previousBounds) {
            const hiddenChildren = sourceNode._children || [];
            if (hiddenChildren.length === 0) {
              update(sourceNode, false, 0);
              applyViewportPolicy(
                sourceNode.parent ? "inner-toggle" : "root-toggle",
                { previousBounds }
              );
              return;
            }
            sourceNode.children = sourceNode._children;
            sourceNode._children = null;
            sourceNode.children.forEach(function(child) {
              collapse(child, 0, 1);
            });
            const layers = getRelativeDepthLayers(sourceNode.children || []);
            const stagedIds = new Set(layers.flat().map((node) => node.id));
            update(sourceNode, false, 0, {
              hiddenNodeIds: stagedIds,
              hiddenLinkIds: stagedIds
            });
            layers.forEach((layer) => {
              const layerIds = new Set(layer.map((node) => node.id));
              selectTreeLinksByIds(layerIds).style("stroke-opacity", 0).style("pointer-events", "none");
              selectTreeNodesByIds(layerIds).style("opacity", 0).style("pointer-events", "none").attr("transform", (node) => getNodeTransform(node, 0.96));
            });
            for (const layer of layers) {
              const layerIds = new Set(layer.map((node) => node.id));
              const layerLinks = selectTreeLinksByIds(layerIds);
              const layerNodes = selectTreeNodesByIds(layerIds);
              await runTransition(
                layerLinks,
                (transition) => transition.duration(stagedToggleDurations.expandLink).ease(d3.easeCubicOut).style("stroke-opacity", 1)
              );
              await runTransition(
                layerNodes,
                (transition) => transition.duration(stagedToggleDurations.expandNode).ease(d3.easeCubicOut).style("opacity", 1).attr("transform", (node) => getNodeTransform(node, 1))
              );
              layerLinks.style("pointer-events", null);
              layerNodes.style("pointer-events", null);
              if (stagedToggleDurations.gap > 0) {
                await wait(stagedToggleDurations.gap);
              }
            }
            treeRoot.descendants().forEach(function(node) {
              node.x0 = node.x;
              node.y0 = node.y;
              node.cx = node.x * xMultiplayer;
              node.cy = node.y * yMultiplayer;
            });
            renderFocusState();
            applyViewportPolicy(
              sourceNode.parent ? "inner-toggle" : "root-toggle",
              { previousBounds }
            );
          }
          async function animateCollapseLayers(sourceNode, previousBounds) {
            const layers = getRelativeDepthLayers(sourceNode.children || []);
            if (layers.length === 0) {
              sourceNode._children = sourceNode.children;
              sourceNode.children = null;
              update(sourceNode, false, 0);
              applyViewportPolicy(
                sourceNode.parent ? "inner-toggle" : "root-toggle",
                { previousBounds }
              );
              return;
            }
            const reversedLayers = layers.slice().reverse();
            for (const layer of reversedLayers) {
              const layerIds = new Set(layer.map((node) => node.id));
              const layerNodes = selectTreeNodesByIds(layerIds);
              const layerLinks = selectTreeLinksByIds(layerIds);
              layerNodes.style("pointer-events", "none");
              layerLinks.style("pointer-events", "none");
              await runTransition(
                layerNodes,
                (transition) => transition.duration(stagedToggleDurations.collapseNode).ease(d3.easeCubicIn).style("opacity", 0).attr("transform", (node) => getNodeTransform(node, 0.96))
              );
              await runTransition(
                layerLinks,
                (transition) => transition.duration(stagedToggleDurations.collapseLink).ease(d3.easeCubicIn).style("stroke-opacity", 0)
              );
              if (stagedToggleDurations.gap > 0) {
                await wait(stagedToggleDurations.gap);
              }
            }
            sourceNode._children = sourceNode.children;
            sourceNode.children = null;
            update(sourceNode, false, 0);
            renderFocusState();
            applyViewportPolicy(
              sourceNode.parent ? "inner-toggle" : "root-toggle",
              { previousBounds }
            );
          }
          async function runStagedToggle(sourceNode, previousBounds) {
            if (sourceNode.children) {
              await animateCollapseLayers(sourceNode, previousBounds);
            } else {
              await animateExpandLayers(sourceNode, previousBounds);
            }
            stLog("debug", sourceNode, "Node w click");
          }
          var startDepth = "$depth";
          if (typeof startDepth == "string") {
            startDepth = 4;
          }
          stLog("debug", startDepth, "start depth value");
          collapse(treeRoot, 0, startDepth);
          update(treeRoot, false, 0);
          applyViewportPolicy("initial");
          if (treeData.model_name != "DecisionTreeClassifier" && treeData.model_name != "DecisionTreeRegressor") {
            stLog("debug", "hello");
            d3.selectAll("#st-info-div-treeID").append("p").text(`${treeData.model_name} ${treeData.which_tree}`).style("font-size", "12px").style("color", "black");
          }
          const mouseover = function(d) {
            tooltipBody.style("opacity", 1);
            tooltipModal.style("opacity", 1);
          };
          const mouseleave = function(d) {
            tooltipBody.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
            tooltipModal.style("opacity", 0).style("top", "-2000px").style("left", "-2000px");
          };
          const mousemoveButton = function(event2, d) {
            tooltipBody.html(`<b>${d}</b>`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
            tooltipModal.html(`<b>${d}</b>`).style("top", event2.pageY - 10 + "px").style("left", event2.pageX + 10 + "px");
          };
          d3.select("#openModalBtn-treeID").on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
            mousemoveButton(event, "Open modal window");
          });
          var toolbar = primaryToolbarGroup;
          var saveSvgbutton = toolbar.append("button").html(svgDownload).attr("id", "svgButton").attr("class", "st-option-button").on("click", saveSvg).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
            mousemoveButton(event, "Save SVG");
          });
          if (treeData.show_sample !== void 0 && treeData.show_sample != "nodata" && !treeData.tree_type.startsWith("nodata")) {
            var showSampleButton = primaryToolbarGroup.append("button").html(svgSample).attr("id", "showSampleButton").attr("class", "st-option-button").on("click", showSample).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
              mousemoveButton(event, "Show sample path");
            });
          }
          if (!treeData.tree_type.startsWith(nodata))
            secondaryToolbarGroup.append("button").html(svgLine).attr("id", "boldLink").attr("class", "st-option-button").on("click", boldClick).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
              mousemoveButton(
                event,
                "Change line tickness scalling in reference to samples in child node"
              );
            });
          d3.selectAll("#st-close-button-treeID").on("click", function() {
            d3.selectAll("#st-side-panel-treeID").classed("show", false).classed("hide", true);
            setTimeout(function() {
              sideSVG.selectAll("g").remove();
            }, 300);
          });
          secondaryToolbarGroup.append("button").html(svgZoom).attr("id", "fitVisible").attr("class", "st-option-button").on("click", () => applyViewportPolicy("fit-visible")).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
            mousemoveButton(event, "Fit visible tree");
          });
          secondaryToolbarGroup.append("button").html(svgFitFull).attr("id", "fitFull").attr("class", "st-option-button").on("click", () => applyViewportPolicy("fit-full")).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
            mousemoveButton(event, "Fit full tree");
          });
          if (treeData.tree_type == classification) {
            tertiaryToolbarGroup.append("button").html(svgXAxis).attr("id", "changeXAxis").attr("class", "st-option-button").on("click", xClick).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
              mousemoveButton(event, "Change Scale on X Axis");
            });
            tertiaryToolbarGroup.append("button").html(svgYAxis).attr("id", "changeYAxis").attr("class", "st-option-button").on("click", yClick).on("mouseover", mouseover).on("mouseleave", mouseleave).on("mousemove", function(d) {
              mousemoveButton(event, "Change Scale on Y Axis");
            });
          }
          const myToolbar = toolbarRoot;
          if (treeData.tree_type == classification && treeData.show_palette_control) {
            let dropdownColors = tertiaryToolbarGroup.append("select").attr("id", "st-color-dropdown").attr("class", "st-dropdown").on("change", function() {
              var number = extractNumber(this.value);
              colors = allColors.slice((number - 1) * colorSize, number * colorSize);
              if (treeData.feature_names.length > colors.length) {
                let getRandomColor = function() {
                  let letters = "0123456789ABCDEF";
                  let color = "#";
                  for (let i = 0; i < 6; i++) {
                    color += letters[Math.floor(Math.random() * 16)];
                  }
                  return color;
                };
                let missingColors = treeData.feature_names.length - colors.length;
                for (let i = 0; i < missingColors; i++) {
                  colors.push(getRandomColor());
                }
              }
              update(treeRoot, true);
            });
            let optionsColors = [];
            d3.selectAll(".st-option-button").style("user-select", "none").style("-webkit-user-select", "none").style("-moz-user-select", "none").style("-ms-user-select", "none");
            for (let i = 1; i <= allColors.length / 20; i++) {
              optionsColors.push(`Palette = ${i}`);
            }
            dropdownColors.selectAll("option").data(optionsColors).enter().append("option").attr("value", (d) => d).text((d) => d);
          }
          setTimeout(function() {
            const logoURL = "https://mljar.com/images/logo/logo_blue_white.svg";
            tertiaryToolbarGroup.append("button").attr("class", "st-option-button").style("background", "transparent").style("border", "none").style("cursor", "pointer").style("padding", "0").style("position", "relative").append("img").attr("src", logoURL).style("height", "50px");
          }, 100);
          const maxDepth = getTreeDepth(treeRoot, 0);
          let currentDepthValue = Math.max(1, Math.min(startDepth, maxDepth));
          const depthControl = depthToolbarGroup.append("div").attr("class", "st-depth-control");
          const depthDecreaseButton = depthControl.append("button").attr("id", "st-depth-decrease-treeID").attr("class", "st-option-button st-depth-button").attr("type", "button").text("-");
          const depthLabel = depthControl.append("div").attr("id", "st-depth-label-treeID").attr("class", "st-depth-label");
          const depthIncreaseButton = depthControl.append("button").attr("id", "st-depth-increase-treeID").attr("class", "st-option-button st-depth-button").attr("type", "button").text("+");
          depthDecreaseButton.on("click", function() {
            applyDepthChange(currentDepthValue - 1);
          });
          depthIncreaseButton.on("click", function() {
            applyDepthChange(currentDepthValue + 1);
          });
          syncDepthControls();
        }
      });
    }).catch((error) => {
      stLog("error", error.message);
    });
  }

  // supertree/js/src/index.js
  buildTree();
})();
