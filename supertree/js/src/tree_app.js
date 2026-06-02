import {
  svgDownload,
  svgFitVisible,
  svgLine,
} from "./icons.js";
import { createTreeLayoutHelpers } from "./layout.js";
import {
  processClassificationLeaf,
  processClassificationNode,
  processRegressionLeaf,
  processRegressionNode,
} from "./node_renderers.js";
import { stLog } from "./shared.js";
import { createViewportController } from "./viewport.js";

export function buildTree(
  pathJson = "data/bugdata.json",
  pytree = "$treetemplate",
) {
  async function checkD3Element(selector, timeout = 5000) {
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
  checkD3Element("#graph-div-treeID")
    .then(() => {
      d3.select("#graph-div-treeID").attr(
        "class",
        "st-body-tree-div st-body-tree-div-treeID",
      );

      let myid = Math.random();
      const logoURL = "https://mljar.com/images/logo/logo_blue_white.svg";

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

      d3.selectAll("#graph-div-treeID")
        .append("div")
        .attr("class", "st-tree-watermark")
        .append("img")
        .attr("src", logoURL)
        .attr("alt", "MLJAR");

      var stjupyter = false;

      if (document.body.getAttribute('data-jp-theme-name') !== null) {
        stjupyter = true;
      }

      const toolbarRoot = d3.select("#toolbar-treeID");
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

      let modal = document.getElementById("myModal-treeID");
      let btn = null;
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
      let depthUnlockTimer = null;
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
        modal: 220,
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
        "#227944",
      ];
      const colorSize = 20;


      var tooltipModal = d3
        .selectAll("#myModal-treeID")
        .append("div")
        .attr("class", "st-tooltip")
        .style("position", "absolute")
        .style("opacity", 0)
        .style("-webkit-user-select", "none")
        .style("-moz-user-select", "none")
        .style("-ms-user-select", "none")
        .style("font-size", "18px")
        .style("background-color", "rgba(0, 0, 0, 0.7)")
        .style("color", "white")
        .style("padding", "8px")
        .style("border-radius", "4px")
        .style("box-shadow", "0px 4px 8px rgba(0, 0, 0, 0.3)")
        .style("max-width", "200px")
        .style("text-align", "center")
        .style("z-index", "1000")
        .style("transition", "opacity 0.3s ease");

      var tooltipBody = d3
        .selectAll("body")
        .append("div")
        .attr("class", "st-tooltip")
        .style("position", "absolute")
        .style("opacity", 0)
        .style("user-select", "none")
        .style("-webkit-user-select", "none")
        .style("-moz-user-select", "none")
        .style("-ms-user-select", "none")
        .style("font-size", "18px")
        .style("background-color", "rgba(0, 0, 0, 0.7)")
        .style("color", "white")
        .style("padding", "8px")
        .style("border-radius", "4px")
        .style("box-shadow", "0px 4px 8px rgba(0, 0, 0, 0.3)")
        .style("max-width", "200px")
        .style("text-align", "center")
        .style("z-index", "1000")
        .style("transition", "opacity 0.3s ease");


          function startWatcher() {
        const treeID = "st-body-tree-div-treeID";
        const modalID = "myModal-treeID";

        function callback(mutationsList, observer) {
            const treeElement = d3.select("." + treeID);
            if (treeElement.empty()) {
                const modalElement = d3.select("#" + modalID);
                if (!modalElement.empty()) {
                    modalElement.remove();
                }
                observer.disconnect();
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
              treeData: treeData.tree_data,
            };
          } else {
            const [treeDataResponse] = await Promise.all([fetch(pathJson)]);

            if (!treeDataResponse.ok) {
              throw new Error("Network response was not ok");
            }

            const treeData = await treeDataResponse.json();
            stLog("debug", treeData);
            data = {
              nodeData: treeData.node_data,
              treeData: treeData.tree_data,
            };
          }
          return data;
        } catch (error) {
          stLog("error", "Error loading JSON files:");
        }
      }

      loadJSONFiles().then((data) => {
        if (data) {
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
            getTreeDepth,
          } = createTreeLayoutHelpers({
            treeDataRoot: nodeData,
            treeLeafSpacing,
            treeDepthSpacing,
            treeLevelSpacing,
            xMultiplayer,
            yMultiplayer,
          });

          const stableLayout = computeStableLayout(treeDataConverted);


          let featureNumber = treeData.data_feature[0].length;
          var globalXExtent = Array.from({ length: featureNumber }, () => [
            Infinity,
            -Infinity,
          ]);
          var globalYExtent = Array.from({ length: featureNumber }, () => [
            0,
            -Infinity,
          ]);
          async function click(event, d) {
            if (isLocked) return;
            if (d.children == null && d._children == null) {
              return;
            }
            stLog("debug", " Collapse click event");
            setControlsLocked(true);
            const previousBounds = d.parent
              ? getVisibleTreeBounds({
                fallbackToFullForSingleRoot: true,
                useRenderedBounds: false,
              })
              : null;
            try {
              await runStagedToggle(d, previousBounds);
            } catch (error) {
              stLog("error", error, "Toggle animation failed");
              setControlsLocked(false);
              throw error;
            }
          }

          let nodeTreeMargin = { top: 20, right: 90, bottom: 160, left: 90 },
            treeWidth = stableLayout.layoutWidth,
            treeHeight = stableLayout.layoutHeight;

          const duration = interactionDurations.toggle;
          const stagedToggleDurations = {
            expandLink: 110,
            expandNode: 140,
            collapseNode: 120,
            collapseLink: 90,
            gap: 14,
          };
          let currentFocusNodeId = null;
          let currentFocusAction = "Ready";
          let currentFocusLabel = "Root";


          const paletteIndex = Math.max((treeData.palette || 1) - 1, 0);
          let colors = allColors.slice(paletteIndex * colorSize, (paletteIndex + 1) * colorSize);

          if (treeData.feature_names.length > colors.length) {
            let missingColors = treeData.feature_names.length - colors.length;

            function getRandomColor() {
              let letters = '0123456789ABCDEF';
              let color = '#';
              for (let i = 0; i < 6; i++) {
                color += letters[Math.floor(Math.random() * 16)];
              }
              return color;
            }

            for (let i = 0; i < missingColors; i++) {
              colors.push(getRandomColor());
            }
          }

          var zoom = d3.zoom().scaleExtent([0.05, 5]).on("zoom", zoomed);

          function zoomed(event) {
            treeSVG.attr("transform", event.transform);
          }

          var treeRoot = createHierarchyFromData();
          applyStableLayout(treeRoot, stableLayout);
          treeRoot.x0 = treeRoot.x;
          treeRoot.y0 = treeRoot.y;

          var treeSVG = d3
            .selectAll("#graph-div-treeID")
            .append("svg")
            .attr(
              "width",
              treeWidth + nodeTreeMargin.right + nodeTreeMargin.left,
            )
            .attr(
              "height",
              treeHeight + nodeTreeMargin.top + nodeTreeMargin.bottom,
            )
            .attr("id", "mySVG-treeID")
            .attr("class", "st-svg")
            .call(zoom)
            .append("g")
            .attr(
              "transform",
              "translate(" +
              nodeTreeMargin.left +
              "," +
              nodeTreeMargin.top +
              ")",
            );

          var sideSVG = d3.selectAll("#st-side-panel-treeID").append("svg")
            .attr("id", "st-side-svg-treeID")
            .attr("class", "st-svg-2")
            .attr("width", "100%")
            .attr("height", 1000)
            .style("background-color", "#f3f9fb");

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
              if (!locked && depthUnlockTimer !== null) {
                clearTimeout(depthUnlockTimer);
                depthUnlockTimer = null;
              }
              if (typeof syncDepthControls === "function") {
                syncDepthControls();
              }
            },
          });
          const setControlsLocked = function(locked) {
            viewportController.setControlsLocked(locked);
            d3.selectAll("#st-depth-decrease-treeID").attr("disabled", locked ? "disabled" : null);
            d3.selectAll("#st-depth-increase-treeID").attr("disabled", locked ? "disabled" : null);
          };
          const applyViewportPolicy = viewportController.applyViewportPolicy;
          const getVisibleTreeBounds = viewportController.getVisibleTreeBounds;

          function scheduleDepthUnlock(delay = interactionDurations.fit + 40) {
            if (depthUnlockTimer !== null) {
              clearTimeout(depthUnlockTimer);
            }
            depthUnlockTimer = setTimeout(function() {
              depthUnlockTimer = null;
              setControlsLocked(false);
            }, delay);
          }

          function getNodeDisplayLabel(node) {
            if (!node) {
              return "Root";
            }
            if (!node.parent) {
              return "Root";
            }
            if (node.data && node.data.is_leaf) {
              return `Leaf ${node.id}`;
            }
            if (
              node.data &&
              Number.isInteger(node.data.feature) &&
              treeData.feature_names &&
              treeData.feature_names[node.data.feature]
            ) {
              return treeData.feature_names[node.data.feature];
            }
            return `Node ${node.id}`;
          }

          function getNodeById(nodeId) {
            return treeRoot.descendants().find((node) => node.id === nodeId) || null;
          }

          function getAncestorIds(node) {
            const ids = new Set();
            let current = node;
            while (current) {
              ids.add(current.id);
              current = current.parent;
            }
            return ids;
          }

          function getDescendantIds(node) {
            const ids = new Set();
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
          }

          function wait(ms) {
            return new Promise((resolve) => setTimeout(resolve, ms));
          }

          function getRelativeDepthLayers(nodes) {
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
          }

          function selectTreeNodesByIds(ids) {
            return treeSVG.selectAll(".treeNode").filter((node) => ids.has(node.id));
          }

          function selectTreeLinksByIds(ids) {
            return treeSVG.selectAll("#st-link-treeID").filter((node) => ids.has(node.id));
          }

          function getNodeTransform(node, scale = 1) {
            return (
              "translate(" +
              node.x * xMultiplayer +
              "," +
              node.y * yMultiplayer +
              ") scale(" +
              scale +
              ")"
            );
          }

          function runTransition(selection, configureTransition) {
            if (selection.size() === 0) {
              return Promise.resolve();
            }

            const transition = configureTransition(selection.transition());
            return transition.end().catch(() => {});
          }

          async function animateExpandLayers(sourceNode, previousBounds) {
            const hiddenChildren = sourceNode._children || [];

            if (hiddenChildren.length === 0) {
              update(sourceNode, false, 0);
              applyViewportPolicy(
                sourceNode.parent ? "inner-toggle" : "root-toggle",
                { previousBounds },
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
              hiddenLinkIds: stagedIds,
            });

            layers.forEach((layer) => {
              const layerIds = new Set(layer.map((node) => node.id));
              selectTreeLinksByIds(layerIds)
                .style("stroke-opacity", 0)
                .style("pointer-events", "none");
              selectTreeNodesByIds(layerIds)
                .style("opacity", 0)
                .style("pointer-events", "none")
                .attr("transform", (node) => getNodeTransform(node, 0.96));
            });

            for (const layer of layers) {
              const layerIds = new Set(layer.map((node) => node.id));
              const layerLinks = selectTreeLinksByIds(layerIds);
              const layerNodes = selectTreeNodesByIds(layerIds);

              await runTransition(layerLinks, (transition) =>
                transition
                  .duration(stagedToggleDurations.expandLink)
                  .ease(d3.easeCubicOut)
                  .style("stroke-opacity", 1),
              );

              await runTransition(layerNodes, (transition) =>
                transition
                  .duration(stagedToggleDurations.expandNode)
                  .ease(d3.easeCubicOut)
                  .style("opacity", 1)
                  .attr("transform", (node) => getNodeTransform(node, 1)),
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
              { previousBounds },
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
                { previousBounds },
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

              await runTransition(layerNodes, (transition) =>
                transition
                  .duration(stagedToggleDurations.collapseNode)
                  .ease(d3.easeCubicIn)
                  .style("opacity", 0)
                  .attr("transform", (node) => getNodeTransform(node, 0.96)),
              );

              await runTransition(layerLinks, (transition) =>
                transition
                  .duration(stagedToggleDurations.collapseLink)
                  .ease(d3.easeCubicIn)
                  .style("stroke-opacity", 0),
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
              { previousBounds },
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

          function renderFocusState() {
            const focusedNode = currentFocusNodeId === null ? null : getNodeById(currentFocusNodeId);
            const activeIds = focusedNode ? getAncestorIds(focusedNode) : new Set();
            const contextIds = focusedNode ? getDescendantIds(focusedNode) : new Set();
            const anyFocus = focusedNode !== null;

            treeSVG
              .selectAll(".treeNode")
              .classed("st-node-active", (d) => anyFocus && d.id === focusedNode.id)
              .classed("st-node-context", (d) => anyFocus && contextIds.has(d.id) && d.id !== focusedNode.id)
              .classed("st-node-path", (d) => anyFocus && activeIds.has(d.id) && d.id !== focusedNode.id)
              .classed("st-node-dim", (d) => anyFocus && !contextIds.has(d.id) && !activeIds.has(d.id));

            treeSVG
              .selectAll("#st-link-treeID")
              .classed("st-link-active", (d) => anyFocus && d.id === focusedNode.id)
              .classed("st-link-context", (d) => anyFocus && contextIds.has(d.id) && d.id !== focusedNode.id)
              .classed("st-link-path", (d) => anyFocus && activeIds.has(d.id) && d.id !== focusedNode.id)
              .classed("st-link-dim", (d) => anyFocus && !contextIds.has(d.id) && !activeIds.has(d.id));

            d3.select("#st-nav-action-treeID").text(currentFocusAction);
            d3.select("#st-nav-focus-treeID").text(currentFocusLabel);
          }

          function setInteractionFocus(node, actionLabel) {
            currentFocusNodeId = node ? node.id : null;
            currentFocusAction = actionLabel;
            currentFocusLabel = getNodeDisplayLabel(node);
            renderFocusState();
          }



          var startDepth = "$depth"

          if (typeof startDepth == "string") {
            startDepth = 4;
          }

          stLog("debug", startDepth, "start depth value");

          collapse(treeRoot, 0, startDepth);
          update(treeRoot, false, 0);
          applyViewportPolicy("initial");

          function update(source, resetTree, transitionDuration = duration, options = {}) {
            const hiddenNodeIds = options.hiddenNodeIds || new Set();
            const hiddenLinkIds = options.hiddenLinkIds || hiddenNodeIds;
            var maxDepthReset = 0;
            if (resetTree) {

              function DFS(node, depth, maxDepth) {
                depth++;
                if (node.children) {
                  node.children.forEach(function(child) {
                    maxDepthReset = Math.max(depth, maxDepthReset);
                    DFS(child, depth, maxDepth);
                  })
                }
              }
              DFS(treeRoot, 0, maxDepthReset);

              treeSVG.selectAll(".treeNode").remove();
              treeSVG.selectAll("#st-link-treeID").remove();
              treeSVG.selectAll("g").remove();
              treeRoot = createHierarchyFromData();
              collapse(treeRoot, 0, maxDepthReset + 1);
              applyStableLayout(treeRoot, stableLayout);
              stLog("debug", treeSVG, "treeSVG")
            }
            d3.selectAll("#st-link-treeID").style("stroke", defaultLinkStroke);
            minSample = Infinity;
            applyStableLayout(treeRoot, stableLayout);
            let links = treeRoot.descendants().slice(1);

            var treeNode = treeSVG
              .selectAll(".treeNode")
              .data(treeRoot.descendants(), function(d) {
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

            let enterDescendants = getDescendants(source);

            let idsArrayDescendants = enterDescendants.map(function(d) {
              return d.id;
            });

            idsArrayDescendants.push(source.id);
            const enteredLinkIds = new Set(
              idsArrayDescendants.filter((id) => id !== source.id),
            );
            const enteringNodeScale = 0.96;
            const enteredNodeRevealDuration = Math.max(
              Math.min(transitionDuration * 0.55, 220),
              0,
            );

            var treeNodeEnter = treeNode
              .enter()
              .append("g")
              .attr("class", "treeNode")
              .style("opacity", function(d) {
                return hiddenNodeIds.has(d.id) ? 0 : 0;
              })
              .attr(
                "transform",
                function(d) {
                  if (hiddenNodeIds.has(d.id)) {
                    return getNodeTransform(d, enteringNodeScale);
                  }

                  return (
                    "translate(" +
                    source.cx +
                    "," +
                    source.cy +
                    ") scale(" +
                    enteringNodeScale +
                    ")"
                  );
                },
              )
              .on("click", click);

            globalXExtent = Array.from({ length: featureNumber }, () => [
              Infinity,
              -Infinity,
            ]);
            globalYExtent = Array.from({ length: featureNumber }, () => [
              0,
              -Infinity,
            ]);


            for (let i = 0; i < featureNumber; i++) {
              let tempArr = [];
              for (let j = 0; j < treeData.data_feature.length; j++) {
                tempArr.push(treeData.data_feature[j][i]);
              }
              let tempGlobalXExtent = d3.extent(tempArr);
              globalXExtent[i][0] = Math.min(
                globalXExtent[i][0],
                tempGlobalXExtent[0],
              );
              globalXExtent[i][1] = Math.max(
                globalXExtent[i][1],
                tempGlobalXExtent[1],
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
                  for (
                    let i = 0;
                    i < d.data.class_distribution[0].length;
                    i++
                  ) {
                    sum = sum + parseFloat(d.data.class_distribution[0][i]);
                  }
                  maxSample = Math.max(maxSample, sum);
                  minSample = Math.min(minSample, sum);
                }
              });
              if (maxSample > 0) {
                globalMaxSample = Math.max(maxSample,globalMaxSample);
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
                  d3.select(this)
                    .append("rect")
                    .attr("class", "histogram-background")
                    .attr("x", -(scatterplotWidth / 2) - 25)
                    .attr("y", -10)
                    .attr("width", rectWidth)
                    .attr("height", rectHeight)
                    .attr("stroke-width", 1)
                    .attr("stroke", "#545454")
                    .attr("rx", 10)
                    .attr("ry", 10)
                    .style("fill", "#ffffff");

                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 10)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Threshold " + parseFloat(d.data.threshold).toFixed(3))
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");

                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 30)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Impurity: " + d.data.impurity)
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");

                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 50)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Samples " + d.data.samples)
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");

                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 70)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Value [" + d.data.class_distribution + "]")
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");

                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 90)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Class " + d.data.treeclass)
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");



                }
                if (d.data.is_leaf) {
                  d3.select(this)
                    .append("rect")
                    .attr("class", "histogram-background")
                    .attr("x", -(scatterplotWidth / 2) - 5)
                    .attr("y", -10)
                    .attr("width", rectWidth - 40)
                    .attr("height", rectHeight)
                    .attr("stroke-width", 1)
                    .attr("stroke", "#545454")
                    .attr("rx", 10)
                    .attr("ry", 10)
                    .style("fill", "#ffffff");


                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 10)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Samples " + d.data.samples)
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");


                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 40)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Value [" + d.data.class_distribution + "]")
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");


                  d3.select(this)
                    .append("text")
                    .attr("dy", ".0em")
                    .attr("y", 70)
                    .attr("class", "st-target")
                    .style("text-anchor", "middle")
                    .text((d) => "Class " + d.data.treeclass)
                    .style("-webkit-user-select", "none")
                    .style("-moz-user-select", "none")
                    .style("-ms-user-select", "none");



                }
              });


            }

            var treeNodeUpdate = treeNodeEnter.merge(treeNode);

            treeNodeUpdate
              .transition()
              .duration(function(d) {
                if (hiddenNodeIds.has(d.id)) {
                  return 0;
                }
                if (enteredLinkIds.has(d.id)) {
                  return enteredNodeRevealDuration;
                }
                return transitionDuration;
              })
              .ease(d3.easeCubicInOut)
              .style("opacity", function(d) {
                return hiddenNodeIds.has(d.id) ? 0 : 1;
              })
              .tween("logging", function(d) {
                let interpolateX = d3.interpolate(
                  source.x0 * xMultiplayer,
                  d.x * xMultiplayer,
                );
                let interpolateY = d3.interpolate(
                  source.y * yMultiplayer,
                  d.y * yMultiplayer,
                );

                return function(t) {
                  let x = interpolateX(t);
                  let y = interpolateY(t);
                  d.cx = x;
                  d.cy = y;
                };
              })
              .attr("transform", function(d) {
                if (hiddenNodeIds.has(d.id)) {
                  return getNodeTransform(d, enteringNodeScale);
                }

                return (
                  "translate(" +
                  d.x * xMultiplayer +
                  "," +
                  d.y * yMultiplayer +
                  ") scale(1)"
                );
              });

            let exitDescendants = [];

            exitDescendants = getDescendants(source);

            let descendantIds = new Set(exitDescendants.map((d) => d.id));

            let treeNodeExit = treeNode
              .exit()
              .filter(function(d) {
                return descendantIds.has(d.id);
              })
              .transition()
              .duration(transitionDuration)
              .ease(d3.easeCubicInOut)
              .attr("transform", function(d) {
                return (
                  "translate(" +
                  (source.x + rectWidth / 2) +
                  "," +
                  (source.y0 + rectHeight / 2) +
                  ")"
                );
              })
              .attr("font-size", "1em")
              .remove();

            treeNodeExit
              .selectAll(".st-target")
              .style("fill-opacity", 1e-6)
              .style("font-size", "0px")
              .attr("x", -rectWidth / 2)
              .attr("y", 0);

            treeNodeExit
              .selectAll(".st-triangle")
              .style("stroke-width", 0)
              .style("fill-opacity", 0)
              .attr("transform", function(d) {
                return "translate(" + (-source.x / 6) + "," + (-source.y0 / 6) + ")";
              })
              .size(0);

            const smallScaleX = d3.scaleLinear().domain(0, 10).range([0, 1]);
            const smallScaleY = d3.scaleLinear().domain(10, 0).range([1, 0]);

            treeNodeExit
              .selectAll(".xAxis")
              .attr("transform", "translate(" + -rectWidth / 2 + "," + 0 + ")")
              .call(
                d3
                  .axisBottom(smallScaleX)
                  .tickSize(0)
                  .tickPadding(8)
                  .ticks(2)
                  .tickFormat(d3.format(",.1f")),
              );

            treeNodeExit
              .selectAll(".yAxis")
              .attr("transform", "translate(" + -rectWidth / 2 + "," + 0 + ")")
              .call(
                d3
                  .axisRight(smallScaleY)
                  .tickSize(0)
                  .tickPadding(8)
                  .ticks(2)
                  .tickFormat(d3.format(",.1f")),
              );

            treeNodeExit
              .selectAll("rect.histogram-background")
              .attr("width", 1e-6)
              .attr("height", 1e-6)
              .attr("x", -rectWidth / 2)
              .attr("y", 0);

            treeNodeExit
              .selectAll("rect.bar")
              .attr("width", 1e-6)
              .attr("height", 1e-6)
              .attr("x", -histogramWidth / 2 + 70)
              .attr("y", 0);

            treeNodeExit
              .selectAll("circle")
              .attr("r", 1e-6)
              .attr("cx", -scatterplotWidth / 2 + 20)
              .attr("cy", 0);

            treeNodeExit
              .selectAll("path.piechart")
              .attr("transform", "translate(" + -rectWidth / 2 + "," + 0 + ")")
              .attr("d", d3.arc().innerRadius(0).outerRadius(1));
            let gridAnimation = treeNodeExit.select(".st-grid");

            treeNodeExit
              .selectAll("line.threshold-line")
              .attr("transform", "translate(" + -rectWidth + "," + -40 + ")")
              .attr("x1", rectWidth / 2)
              .attr("x2", rectWidth / 2)
              .attr("y1", rectHeight / 2 - 15)
              .attr("y2", rectHeight / 2)
              .attr("stroke-width", 0);

            treeNodeExit
              .selectAll("line.average-line")
              .attr("transform", "translate(" + -rectWidth + "," + -40 + ")")
              .attr("x1", rectWidth / 2)
              .attr("x2", rectWidth / 2)
              .attr("y1", rectHeight / 2 - 15)
              .attr("y2", rectHeight / 2 - 15)
              .attr("stroke-width", 0);

            gridAnimation.selectAll("line").attr("x2", 0).attr("y2", 0);

            treeNodeExit
              .selectAll(".yAxis-text")
              .attr("transform", "translate(" + 0 + "," + -40 + ")")
              .style("fill-opacity", 1e-6)
              .style("font-size", "0px")
              .attr("x", 0);

            treeNodeExit
              .selectAll(".xAxis-text")
              .attr("transform", "translate(" + 0 + "," + -40 + ")")
              .style("fill-opacity", 1e-6)
              .style("font-size", "0px");

            treeNodeExit
              .selectAll(".st-pie-target")
              .attr("y", function(d, i) {
                const radius = Math.max(
                  20,
                  (Math.min(pieWidth, pieHeight) / 2) *
                  Math.sqrt(d.data.classDistributionValue / globalMaxSample),
                );

                return -10 - radius * 2 - i * 40;
              })
              .attr("transform", "translate(" + -rectWidth / 2 + "," + 0 + ")")
              .style("fill-opacity", 1e-6)
              .style("font-size", "0px");

            treeNodeExit
              .selectAll(".st-pie-target2")
              .attr("y", function(d, i) {
                const radius = Math.max(
                  20,
                  (Math.min(pieWidth, pieHeight) / 2) *
                  Math.sqrt(d.data.classDistributionValue / globalMaxSample),
                );

                return -10 - radius * 2 - i * 40;
              })
              .attr("transform", "translate(" + -rectWidth / 2 + "," + 0 + ")")
              .style("fill-opacity", 1e-6)
              .style("font-size", "0px");

            let link = treeSVG.selectAll("#st-link-treeID").data(links, function(d) {
              return d.id;
            });

            const mouseover = function(d) {

              const currentStroke = d3.select(this).style("stroke");
              this.dataset.previousStroke = currentStroke;

              tooltipBody.style("opacity", 1);
              tooltipModal.style("opacity", 1);

              d3.select(this).style("stroke", "#0099cc");
            };

            const mouseleave = function(d) {
              const previousStroke = this.dataset.previousStroke || defaultLinkStroke;

              tooltipBody
                .style("opacity", 0)
                .style("top", -2000 + "px")
                .style("left", -2000 + "px");

              tooltipModal
                .style("opacity", 0)
                .style("top", -2000 + "px")
                .style("left", -2000 + "px");

              d3.select(this).style("stroke", previousStroke);
              delete this.dataset.previousStroke;
            };

            const mousemove = function(event, d) {
              if (treeData.tree_type == classification) {
                var currentDistribution = 0;
                for (let i = 0; i < d.data.class_distribution[0].length; i++) {
                  currentDistribution =
                    currentDistribution +
                    parseInt(d.data.class_distribution[0][i]);
                }

                tooltipModal
                  .html(
                    `<b>Samples in link:</b> ${currentDistribution}`,
                  )
                  .style("top", event.pageY - 10 + "px")
                  .style("left", event.pageX + 10 + "px");

                tooltipBody
                  .html(
                    `<b>Samples in link:</b> ${currentDistribution}`,
                  )
                  .style("top", event.pageY - 10 + "px")
                  .style("left", event.pageX + 10 + "px");
              }
              if (treeData.tree_type == regr) {
                var currentDistribution = d.data.samples;

                tooltipModal
                  .html(`<b>Samples in link:</b> ${currentDistribution}`)
                  .style("top", event.pageY - 10 + "px")
                  .style("left", event.pageX + 10 + "px");

                tooltipBody
                  .html(`<b>Samples in link:</b> ${currentDistribution}`)
                  .style("top", event.pageY - 10 + "px")
                  .style("left", event.pageX + 10 + "px");
              }
            };

            var linkEnter = link
              .enter()
              .insert("path", "g")
              .attr("class", "st-link")
              .attr("id", "st-link-treeID")
              .attr("d", function(d) {
                if (hiddenLinkIds.has(d.id)) {
                  return diagonal(getSourceAnchor(d.parent, d), getTargetAnchor(d));
                }
                var o = getSourceAnchor(source, d);
                return diagonal(o, o);
              })
              .style("fill", "none") // Ustawienie fill inline
              .style("stroke", defaultLinkStroke) // Ustawienie stroke inline
              .style("stroke-width", "2px")
              .style("stroke-opacity", function(d) {
                return hiddenLinkIds.has(d.id) ? 0 : 0;
              })
              .on("mouseover", mouseover)
              .on("mouseleave", mouseleave)
              .on("mousemove", mousemove);

            var allSamples = 0;
            var allSamplesRegr = 0;
            if (treeData.tree_type == regr) {
              allSamplesRegr = nodeData.samples;
            }
            for (let i = 0; i < nodeData.class_distribution[0].length; i++) {
              allSamples =
                allSamples + parseInt(nodeData.class_distribution[0][i]);
            }
            if (boldLinks == true) {
              if (treeData.tree_type == classification) {
                linkEnter.each(function(d) {
                  var currentDistribution = 0;
                  for (
                    let i = 0;
                    i < d.data.class_distribution[0].length;
                    i++
                  ) {
                    currentDistribution =
                      currentDistribution +
                      parseInt(d.data.class_distribution[0][i]);
                  }
                  d3.select(this).style(
                    "stroke-width",
                    Math.max(20 * (currentDistribution / allSamples), 1),
                  );
                });
              }
              if (treeData.tree_type == regr) {
                linkEnter.each(function(d) {
                  var currentDistribution = 0;
                  const currentSamples = d.data.samples;
                  d3.select(this).style(
                    "stroke-width",
                    Math.max(20 * (currentSamples / allSamplesRegr), 1),
                  );
                });
              }
            }

            var linkUpdate = linkEnter.merge(link);
            const linkEnterDelay = enteredNodeRevealDuration;

            linkUpdate
              .transition()
              .delay(function(d) {
                if (hiddenLinkIds.has(d.id)) {
                  return 0;
                }
                return enteredLinkIds.has(d.id) ? linkEnterDelay : 0;
              })
              .duration(function(d) {
                if (hiddenLinkIds.has(d.id)) {
                  return 0;
                }
                if (!enteredLinkIds.has(d.id)) {
                  return transitionDuration;
                }
                return Math.max(transitionDuration - linkEnterDelay, 0);
              })
              .ease(d3.easeCubicInOut)
              .style("stroke-opacity", function(d) {
                return hiddenLinkIds.has(d.id) ? 0 : 1;
              })
              .attrTween("d", function(d) {
                if (hiddenLinkIds.has(d.id)) {
                  const hiddenPath = diagonal(
                    getSourceAnchor(d.parent, d),
                    getTargetAnchor(d),
                  );
                  return function() {
                    return hiddenPath;
                  };
                }

                if (enteredLinkIds.has(d.id)) {
                  const settledPath = diagonal(
                    getSourceAnchor(d.parent, d),
                    getTargetAnchor(d),
                  );
                  return function() {
                    return settledPath;
                  };
                }

                return function() {
                  return diagonal(
                    getSourceAnchor(d.parent, d, { useCurrentPosition: true }),
                    getTargetAnchor(d, { useCurrentPosition: true }),
                  );
                };
              });

            var linkExit = link
              .exit()
              .filter(function(d) {
                return descendantIds.has(d.id);
              })
              .transition()
              .duration(transitionDuration)
              .ease(d3.easeCubicInOut)
              .style("stroke-opacity", 0)
              .attr("d", function(d) {
                var o = getSourceAnchor(source, d, { useExitPosition: true });
                return diagonal(o, o);
              })
              .remove();


            function getNodeBasePosition(node, options = {}) {
              if (options.useCurrentPosition) {
                return {
                  x: node.cx,
                  y: node.cy,
                };
              }

              if (options.useExitPosition) {
                return {
                  x: node.x * xMultiplayer,
                  y: node.y0 * yMultiplayer,
                };
              }

              return {
                x: node.x * xMultiplayer,
                y: node.y * yMultiplayer,
              };
            }

            function getSourceAnchor(node, childNode = null, options = {}) {
              const base = getNodeBasePosition(node, options);
              const cornerInset = 16;
              let anchorX = base.x;

              if (childNode) {
                const childBase = getNodeBasePosition(childNode, options);
                if (childBase.x < base.x) {
                  anchorX = base.x - rectWidth / 2 + cornerInset;
                } else if (childBase.x > base.x) {
                  anchorX = base.x + rectWidth / 2 - cornerInset;
                }
              }

              return {
                x: anchorX,
                y: base.y + rectHeight - 10,
              };
            }

            function getTargetAnchor(node, options = {}) {
              const base = getNodeBasePosition(node, options);

              if (node.data && node.data.is_leaf) {
                if (treeData.tree_type == classification) {
                  return {
                    x: base.x + 10,
                    y: base.y - 3,
                  };
                }

                return {
                  x: base.x + 15,
                  y: base.y + 4,
                };
              }

              return {
                x: base.x,
                y: base.y - 10,
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

          }
          if (treeData.model_name != "DecisionTreeClassifier" && treeData.model_name != "DecisionTreeRegressor") {
            stLog("debug","hello");
            d3.selectAll("#st-info-div-treeID")
              .append("p")
              .text(`${treeData.model_name} ${treeData.which_tree}`)
              .style("font-size", "12px")
              .style("color", "black");
          }
          const mouseover = function(d) {
            tooltipBody.style("opacity", 1);
            tooltipModal.style("opacity", 1);
          };

          const mouseleave = function(d) {
            tooltipBody
              .style("opacity", 0)
              .style("top", -2000 + "px")
              .style("left", -2000 + "px");
            tooltipModal
              .style("opacity", 0)
              .style("top", -2000 + "px")
              .style("left", -2000 + "px");
          };

          const mousemoveButton = function(event, d) {
            tooltipBody
              .html(`<b>${d}</b>`)
              .style("top", event.pageY - 10 + "px")
              .style("left", event.pageX + 10 + "px");

            tooltipModal
              .html(`<b>${d}</b>`)
              .style("top", event.pageY - 10 + "px")
              .style("left", event.pageX + 10 + "px");
          };

          var toolbar = primaryToolbarGroup;

          var saveSvgbutton = toolbar
            .append("button")
            .html(svgDownload)
            .attr("id", "svgButton")
            .attr("class", "st-option-button")
            .on("click", saveSvg)
            .on("mouseover", mouseover)
            .on("mouseleave", mouseleave)
            .on("mousemove", function(d) {
              mousemoveButton(event, "Save SVG");
            });

          saveSvgbutton
            .append("span")
            .attr("class", "st-button-label")
            .text("Save SVG");

          function showSample() {
            var sampleNode = null;
            showSampleDFS(treeRoot);
            function showSampleDFS(node) {
              if (node.children) {
                node.children.forEach(function(child) {
                  showSampleDFS(child)
                })
              }
              else if (node._children) {
                node._children.forEach(function(child) {
                  showSampleDFS(child)
                })
              }
              else {
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
              return
            };
            setControlsLocked(true);
            currentDepthValue = sampleNode.depth + 1;
            showDepth(treeRoot, 0, currentDepthValue, true);
            update(treeRoot, false, 0);
            syncDepthControls();
            showpath(sampleNode);
            applyViewportPolicy("sample-path");


          }

          function showpath(d) {
            d3.selectAll("#st-link-treeID").style("stroke", defaultLinkStroke);
            const pathData = getLinksIds(d);
            const ids = pathData.ids;
            const nodedata = pathData.nodedata;



            var newHeight = nodedata.length * 250;
            newHeight = Math.max(newHeight, 1000);

            sideSVG.attr("height", newHeight);

            d3.selectAll("#st-link-treeID")
              .filter(function(d) {
                return ids.includes(d.id);
              })
              .style("stroke", "#0099cc");

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
          }

          function renderNodeData(nodedata) {
            nodedata.slice().reverse().forEach(function(node, i) {
              let translateX = 150;
              let translateY = 200 * i + 100;

              if (treeData.tree_type == regr && node.data.is_leaf) {
                translateX -= 10; 
              }

              const group = sideSVG.append("g")
                .attr("class", "node-group")
                .attr("transform", `translate(${translateX}, ${translateY})`)
                .datum(node);

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

          }



          function getLinksIds(node) {
            var ids = []
            var nodedata = []
            function linkDFS(node) {
              stLog("debug", node, "Node w dfsie");
              ids.push(node.id)
              nodedata.push(node)
              if (node.parent) {
                linkDFS(node.parent);
              }
            }
            linkDFS(node)
            let pathData = {
              ids: ids,
              nodedata: nodedata
            }
            return pathData;
          }



          if (!treeData.tree_type.startsWith(nodata))
            primaryToolbarGroup
              .append("button")
              .html(svgLine)
              .attr("id", "boldLink")
              .attr("class", "st-option-button")
              .on("click", boldClick)
              .on("mouseover", mouseover)
              .on("mouseleave", mouseleave)
              .on("mousemove", function(d) {
                mousemoveButton(
                  event,
                  "Change line tickness scalling in reference to samples in child node",
                );
              });

          if (!treeData.tree_type.startsWith(nodata)) {
            d3.select("#boldLink")
              .append("span")
              .attr("class", "st-button-label")
              .text("Line Width");
          }


          d3.selectAll("#st-close-button-treeID").on("click", function() {
            d3.selectAll("#st-side-panel-treeID").classed("show", false).classed("hide", true);
            setTimeout(function() {
              sideSVG.selectAll("g").remove();
            }, 300);
          });

          primaryToolbarGroup
            .append("button")
            .html(svgFitVisible)
            .attr("id", "fitVisible")
            .attr("class", "st-option-button")
            .on("click", () => applyViewportPolicy("fit-visible"))
            .on("mouseover", mouseover)
            .on("mouseleave", mouseleave)
            .on("mousemove", function(d) {
              mousemoveButton(event, "Fit visible tree");
            });

          d3.select("#fitVisible")
            .append("span")
            .attr("class", "st-button-label")
            .text("Fit Tree");



          function boldClick() {
            boldLinks = !boldLinks;
            update(treeRoot, true);
          }

          function saveSvg() {
            var svgElement = document.getElementById("mySVG-treeID");

            var serializer = new XMLSerializer();
            var svgString = serializer.serializeToString(svgElement);

            var blob = new Blob([svgString], {
              type: "image/svg+xml;charset=utf-8",
            });

            var downloadLink = document.createElement("a");
            downloadLink.href = URL.createObjectURL(blob);
            downloadLink.download = "myDiagram.svg";

            document.body.appendChild(downloadLink);
            downloadLink.click();

            document.body.removeChild(downloadLink);
          }

          const myToolbar = toolbarRoot;


          if (treeData.tree_type == classification && treeData.show_palette_control) {
            let dropdownColors = tertiaryToolbarGroup
              .append("select")
              .attr("id", "st-color-dropdown")
              .attr("class", "st-dropdown")
              .on("change", function() {
                var number = extractNumber(this.value);
                colors = allColors.slice((number - 1) * colorSize, number * colorSize);
                if (treeData.feature_names.length > colors.length) {
                  let missingColors = treeData.feature_names.length - colors.length;

                  function getRandomColor() {
                    let letters = '0123456789ABCDEF';
                    let color = '#';
                    for (let i = 0; i < 6; i++) {
                      color += letters[Math.floor(Math.random() * 16)];
                    }
                    return color;
                  }

                  for (let i = 0; i < missingColors; i++) {
                    colors.push(getRandomColor());
                  }
                }
                update(treeRoot, true);
              });



            let optionsColors = [];

            d3.selectAll(".st-option-button")
              .style("user-select", "none")
              .style("-webkit-user-select", "none")
              .style("-moz-user-select", "none")
              .style("-ms-user-select", "none");

            for (let i = 1; i <= allColors.length / 20; i++) {
              optionsColors.push(`Palette = ${i}`);
            }

            dropdownColors
              .selectAll("option")
              .data(optionsColors)
              .enter()
              .append("option")
              .attr("value", (d) => d)
              .text((d) => d);
          }

          const maxDepth = getTreeDepth(treeRoot, 0);
          let currentDepthValue = Math.max(1, Math.min(startDepth, maxDepth));

          const depthControl = depthToolbarGroup
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

          function syncDepthControls() {
            depthLabel.text(`Depth=${Math.max(currentDepthValue - 1, 0)}`);
            depthDecreaseButton.attr(
              "disabled",
              isLocked || currentDepthValue <= 1 ? "disabled" : null,
            );
            depthIncreaseButton.attr(
              "disabled",
              isLocked || currentDepthValue >= maxDepth ? "disabled" : null,
            );
          }

          function applyDepthChange(depth) {
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
            scheduleDepthUnlock();
          }

          function handleDepthChange(event, optDepth = "optional") {
            if (optDepth !== "optional") {
              applyDepthChange(optDepth);
              return;
            }

            const depth = extractNumber(this.value);
            if (depth != null) {
              applyDepthChange(depth + 1);
            }
          }

          depthDecreaseButton.on("click", function() {
            applyDepthChange(currentDepthValue - 1);
          });

          depthIncreaseButton.on("click", function() {
            applyDepthChange(currentDepthValue + 1);
          });

          syncDepthControls();
          function extractNumber(str) {
            const match = str.match(/\d+/);

            if (match) {
              return parseInt(match[0], 10);
            }

            return null;
          }

        }
      });
    })
    .catch((error) => {
      stLog("error", error.message);
    });
}
