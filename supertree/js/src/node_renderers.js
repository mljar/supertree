import { stLog, yAxisMargin } from "./shared.js";

export function processClassificationNode(treeData, tooltipBody, tooltipModal, globalX, globalXExtent, globalY, globalYExtent, click, histogramWidth, histogramHeight, rectWidth, rectHeight, colors, d) {
  var isSampleExist = false;
  if (!d.data.is_leaf) {
    if (treeData.show_sample != "nodata" && treeData.show_sample != undefined) {
      isSampleExist = true;
    }
    var isSampleExistInThisNode = true;
    if (isSampleExist) {
      isSampleExistInThisNode = true;
    }
    else {
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

    var xExtent = [Infinity, -Infinity];

    d.data.start_end_x_axis.forEach((currentElement, index) => {
      if (currentElement[0] != "notexist") {
        filteredData.forEach((twoDimArray, i) => {
          filteredData[i] = twoDimArray.filter(
            (rowArray) => rowArray[index] < currentElement[0],
          );
        });
      }
      if (currentElement[1] != "notexist") {
        filteredData.forEach((twoDimArray, i) => {
          filteredData[i] = twoDimArray.filter(
            (rowArray) => rowArray[index] > currentElement[1],
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

    let featureData = filteredData.map((subArray) =>
      subArray.map((innerArray) => innerArray[featureIndex]),
    );

    const removedIndices = [];

    let indicesArray = Array.from(
      { length: featureData.length + removedIndices.length },
      (_, i) => i,
    );

    featureData = featureData
      .map((twoDimArray, index) => {
        const filteredArray = twoDimArray.filter(
          (value) => value !== undefined && value !== null,
        );
        if (filteredArray.length === 0) {
          removedIndices.push(index);
        }
        return filteredArray;
      })
      .filter((twoDimArray) => twoDimArray.length > 0);

    let removedSet = new Set(removedIndices);

    indicesArray = indicesArray.filter(
      (value) => !removedSet.has(value),
    );

    featureData.map((currentValue) => {
      var tempxExtent = d3.extent(currentValue);
      xExtent[0] = Math.min(xExtent[0], tempxExtent[0]);
      xExtent[1] = Math.max(xExtent[1], tempxExtent[1]);
    });

    xExtent[0] = xExtent[0] - 0.2;
    xExtent[1] = xExtent[1] + 0.2;

    if (globalX) {
      xExtent = globalXExtent[featureIndex];
    }

    if (xExtent[0] > d.data.threshold) {
      xExtent[0] = d.data.threshold - 0.2
    }
    if (xExtent[1] < d.data.threshold) {
      xExtent[1] = d.data.threshold + 0.2
    }

  if(isSampleExistInThisNode){
    if (xExtent[0] > treeData.show_sample[featureIndex]) {
      xExtent[0] = treeData.show_sample[featureIndex] - 0.2
    }
    if (xExtent[1] < treeData.show_sample[featureIndex]) {
      xExtent[1] = treeData.show_sample[featureIndex] + 0.2
    }
    }

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([0, histogramWidth]);
    d3.select(this)
      .append("rect")
      .attr("class", "histogram-background")
      .attr("x", -(histogramWidth / 2) - 25)
      .attr("y", -10)
      .attr("width", rectWidth)
      .attr("height", rectHeight)
      .attr("stroke-width", 1)
      .attr("stroke", "#545454")
      .attr("rx", 10)
      .attr("ry", 10)
      .style("fill", "#ffffff")
      .on("click", click);

    const mousemoveAllData = function(event, d) {
      stLog("debug", d, "mousemoveAllData")
      tooltipBody
        .html(
          `<b>All Data</b>:<br>Class distribution: ${d.data.class_distribution}
              <br>Impurity: ${d.data.impurity}
              <br>Samples: ${d.data.samples}
              <br>Threshold: ${parseFloat(d.data.threshold).toFixed(3)}
              <br>Treeclass: ${d.data.treeclass}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
        .style("opacity", 1);

      tooltipModal
        .html(
          `<b>All Data</b>:<br>Class distribution: ${d.data.class_distribution}
              <br>Impurity: ${d.data.impurity}
              <br>Samples: ${d.data.samples}
              <br>Threshold: ${parseFloat(d.data.threshold).toFixed(3)}
              <br>Treeclass: ${d.data.treeclass}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
        .style("opacity", 1);
    };

    const mouseleaveAllData = function(event, d) {
      tooltipBody.style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      tooltipModal.style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

    };

    stLog("debug", this, "This");

    d3.select(this)
      .append("text")
      .attr("class", "st-target")
      .attr("x", 0)
      .attr("y", rectHeight + 15)
      .style("text-anchor", "middle")
      .style("font-size", "18px")
      .text(treeData.feature_names[featureIndex])
      .on("mousemove", mousemoveAllData)
      .on("mouseleave", mouseleaveAllData)
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none");

    const xDomain = xScale.domain();
    const xTickValues = [
      xDomain[0],
      d.data.threshold,
      xDomain[1],
    ];

    d3.select(this)
      .append("g")
      .attr("class", "xAxis")
      .attr(
        "transform",
        `translate(${-histogramWidth / 2}, ${histogramHeight})`,
      )

      .call(
        d3
          .axisBottom(xScale)
          .tickSize(0)
          .tickPadding(8)
          .tickValues(xTickValues)
          .tickFormat(d3.format(",.1f")),
      )
      .selectAll(".tick")
      .attr("class", "xAxis-text")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none")
      .style("fill", "black");


    d3.select(this).selectAll(".domain")
      .style("stroke", "black");

    const yScale = d3.scaleLinear().range([histogramHeight, 0]);

    const yAxis = d3.select(this).append("g");

    /*
  const GridLine = () => d3.axisLeft().scale(yScale);
  d3.select(this)
    .append("g")
    .attr("class", "grid")
    .attr("transform", `translate(${-histogramWidth / 2},0)`)
    .call(
      GridLine()
        .tickSize(-histogramWidth, 0, 0)
        .tickFormat("")
        .ticks(10),
    );
*/
    const histogram = d3
      .bin()
      .domain(xScale.domain())
      .thresholds(xScale.ticks(20));

    const binsData = featureData.map((data) => histogram(data));

    stLog("debug", binsData, "binsData")

    const stackedData = d3
      .stack()
      .keys(d3.range(featureData.length))
      .value((d, key) => (d[key] ? d[key].length : 0))(
        d3.transpose(binsData),
      );



    var yExtent = [0, 0];
    yExtent[0] = 0;
    yExtent[1] = d3.max(stackedData, (d) =>
      d3.max(d, (d) => d[1]),
    );

    globalYExtent[featureIndex][1] = Math.max(
      globalYExtent[featureIndex][1],
      d3.max(stackedData, (d) => d3.max(d, (d) => d[1])),
    );
    if (globalY) {
      yExtent = globalYExtent[featureIndex];
    }
    yScale.domain(yExtent);

    const yDomain = yScale.domain();
    const yTickValues = [
      yDomain[0],
      yDomain[0] + (yDomain[1] - yDomain[0]) / 2,
      yDomain[1],
    ];
    stLog("debug", yTickValues, "YTICKVALUES")
    if (yTickValues.every(value => !isNaN(value))) {
      yAxis
        .call(
          d3
            .axisRight(yScale)
            .tickSize(0)
            .tickPadding(4)
            .tickValues(yTickValues)
            .tickFormat(d3.format(",.0f")),
        )
        .attr("class", "yAxis")
        .attr("transform", `translate(${-histogramWidth / 2 - yAxisMargin},0)`)
        .call((d) => d.select(".domain").remove())
        .style("user-select", "none")
        .style("-webkit-user-select", "none")
        .style("-moz-user-select", "none")
        .style("-ms-user-select", "none")
        .style("fill", "black");
    }


    d3.select(this).selectAll(".domain")
      .style("stroke", "black");

    const mouseover = function(d) {
      tooltipModal.style("opacity", 1);
      tooltipBody.style("opacity", 1);

      d3.select(this).style("stroke", "#EF4A60");
    };

    const mouseleave = function(d) {
      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("stroke", "black");
    };

    const mousemove = function(event, d) {
      tooltipBody
        .html(
          `<b>${treeData.target_names[d.class]}</b>: ${d[1] - d[0]}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(
          `<b>${treeData.target_names[d.class]}</b>: ${d[1] - d[0]}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    let nodeToClick = d;

    stLog("debug", stackedData, "stackedData");
    stLog("debug", d.data, "node:");

    d3.select(this)
      .selectAll("g.layer")
      .data(stackedData)
      .enter()
      .append("g")
      .attr("class", "layer")
      .style("fill", (d, i) => colors[indicesArray[i]])
      .on("click", function() {
        click(source, nodeToClick);
      })
      .selectAll("rect.bar")
      .data((d, i) =>
        d.map((item) => ({ ...item, class: indicesArray[i] })),
      )
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d) => {
        return xScale(d.data[0].x0);
      })
      .attr("y", (d) => {
        if (isNaN(yScale(d[1]))) {
          return 0;
        }
        else {
          return yScale(d[1]);
        }
      })
      .attr("height", (d) => {
        if (isNaN(yScale(d[0]) - yScale(d[1]))) {
          return 0;
        }
        else {
          return yScale(d[0]) - yScale(d[1]);
        }
      })
      .attr(
        "width",
        (d) => xScale(d.data[0].x1) - xScale(d.data[0].x0),
      )
      .attr("transform", `translate(${-histogramWidth / 2},0)`)
      .attr("stroke", "black")
      .on("mouseover", mouseover)
      .on("mouseleave", mouseleave)
      .on("mousemove", mousemove);
    stLog("debug", "abc")
    var threshold = parseFloat(d.data.threshold).toFixed(3);
    d3.select(this)
      .append("line")
      .attr("class", "threshold-line")
      .attr("x1", xScale(threshold))
      .attr("x2", xScale(threshold))
      .attr("y1", 0)
      .attr("y2", histogramHeight)
      .attr("stroke", "black")
      .attr("transform", `translate(${-histogramWidth / 2},0)`)
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5");

    var mouseovertriangle = function(d) {


      tooltipBody.style("opacity", 1);
      tooltipModal.style("opacity", 1);

      d3.select(this).style("fill", "#0099cc").style("stroke", "#0099cc");
    };

    var mouseleavetriangle = function(d) {


      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("fill", "green").style("stroke", "green");
    };


    var mousemovetriangle = function(event, d) {

      tooltipModal
        .html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : 'N/A';
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipBody
        .html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : 'N/A';
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };
    var color = "green";
    var triangleSize = 25;
    var verticalTransform = histogramHeight - Math.sqrt(triangleSize) + 15;

    var triangle = d3.symbol()
      .type(d3.symbolTriangle)
      .size(triangleSize);

    if (isSampleExistInThisNode) {
      stLog("debug", d, "Exist");
      d3.select(this).append("path")
        .attr("d", triangle)
        .attr("class", "st-triangle")
        .style("fill", color)
        .style("stroke-width", 1)
        .style("stroke-opacity", 1)
        .attr("transform", function(d) {
          return "translate(" + (-histogramWidth / 2 + xScale(treeData.show_sample[featureIndex])) + "," + verticalTransform + ")";
        })
        .on("mouseover", mouseovertriangle)
        .on("mouseleave", mouseleavetriangle)
        .on("mousemove", mousemovetriangle);
    }



  }

}


export function processClassificationLeaf(
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
) {
  var formatNumber = d3.format(",.0f");


  if (d.data.is_leaf) {

    const featureIndex = d.data.feature;

    if (treeData.show_sample != "nodata") {
      stLog("debug", treeData.show_sample, "treedatasample");
      isSampleExist = true;
    }
    var isSampleExistInThisNode = true;
    if (isSampleExist) {
      isSampleExistInThisNode = true;
    }
    else {
      isSampleExistInThisNode = false;
    }
    const classDistribution = d.data.class_distribution[0];
    const removedIndexes = [];
    const data = classDistribution
      .map((value, index) => {
        return {
          target_name: `${treeData.target_names[index]}`,
          classDistributionValue: value,
          index: index,
        };
      })
      .filter((item) => {
        if (item.classDistributionValue === 0) {
          removedIndexes.push(item.index);
          return false;
        }
        return true;
      })
      .map((item) => {
        return {
          target_name: item.target_name,
          classDistributionValue: item.classDistributionValue,
        };
      });
    var isSampleExist = false;


    if (treeData.show_sample != "nodata" && treeData.show_sample !== undefined) {
      stLog("debug", treeData.show_sample, "treedatasample");
      isSampleExist = true;
    }
    var isSampleExistInThisNode = true;
    if (isSampleExist) {
      isSampleExistInThisNode = true;
    }
    else {
      isSampleExistInThisNode = false;
    }

    let indicesArray = Array.from(
      { length: classDistribution.length },
      (_, i) => i,
    );

    let removedSet = new Set(removedIndexes);

    indicesArray = indicesArray.filter(
      (value) => !removedSet.has(value),
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

    for (
      let i = 0;
      i < d.data.class_distribution[0].length;
      i++
    ) {
      allCurrentSamples =
        allCurrentSamples +
        parseInt(d.data.class_distribution[0][i]);
    }

    const radius = Math.max(
      20,
      (Math.min(pieWidth, pieHeight) / 2) *
      Math.sqrt(allCurrentSamples / maxSample),
    );
    stLog("debug", allCurrentSamples,"current sample");
    stLog("debug", maxSample,"max sample");

    const pie = d3
      .pie()
      .value((d) => d.classDistributionValue)
      .sort(null);

    const dataPrepared = pie(data);

    const arc = d3.arc().innerRadius(0).outerRadius(radius);

    stLog("debug", "treeID","tree id");
    stLog("debug",radius, "srednicaaaaaaaa");
    stLog("debug",allCurrentSamples, "current samples");
    stLog("debug",maxSample, "max sample");

    data.forEach(function(d) {
      d.classDistributionValue = +d.classDistributionValue;
      d.enabled = true;
    });

    const total = d3.sum(
      data.map(function(d) {
        return d.enabled ? d.classDistributionValue : 0;
      }),
    );

    const mouseover = function(d) {
      tooltipBody.style("opacity", 1);
      tooltipModal.style("opacity", 1);
      d3.select(this).style("stroke", "#0099cc");
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

      d3.select(this).style("stroke", "#545454");
    };

    const mousemove = function(event, d) {
      tooltipBody
        .html(
          `<b>${d.data.target_name}</b>: ${d.data.classDistributionValue}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(
          `<b>${d.data.target_name}</b>: ${d.data.classDistributionValue}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    if (dataPrepared[0] && 'data' in dataPrepared[0]) {
      let nodeToClick = d3.select(this).datum()
      d3.select(this)
        .selectAll("path")
        .data(dataPrepared)
        .join("path")
        .attr("class", "piechart")
        .attr("d", arc)
        .attr("fill", (d, i) => colors[indicesArray[i]])
        .attr("transform", `translate(${10},${radius-2})`)
        .attr("stroke", "#545454")
        .on("mouseover", mouseover)
        .on("mouseleave", mouseleave)
        .on("mousemove", mousemove)
        .on("click", function() {
          showpath(nodeToClick)
        })
        .style("stroke-width", "0.75px")
        .each(function(d) {
          this._current = d;
        })
      stLog("debug", isSampleExistInThisNode, "Exist Sample true false");


      d3.select(this)
        .append("g")
        .attr("class", "st-text-pie")
        .attr("text-anchor", "middle")
        .selectAll(".st-text-pie")
        .data(dataPrepared)
        .join("g")
        .attr(
          "transform",
          (d, i) => `translate(10,${radius * 2 + 20 + i * 40})`,
        )
        .each(function(d, i) {
          const group = d3.select(this);

          group
            .append("text")
            .attr("class", "st-pie-target")
            .attr("x", 0)
            .attr("y", 0)
            .attr("fill", "black")
            .style("text-anchor", "middle")
            .style("font-size", "18px")
            .text(d.data.target_name)
            .style("user-select", "none")
            .style("-webkit-user-select", "none")
            .style("-moz-user-select", "none")
            .style("-ms-user-select", "none");


          group
            .append("text")
            .attr("class", "st-pie-target2")
            .attr("x", 0)
            .attr("y", 20)
            .attr("fill", "black")
            .style("text-anchor", "middle")
            .style("font-size", "18px")
            .text(formatNumber(d.data.classDistributionValue))
            .style("user-select", "none")
            .style("-webkit-user-select", "none")
            .style("-moz-user-select", "none")
            .style("-ms-user-select", "none");

        });
    }
    else {

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
  }
}

export function processRegressionNode(
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
) {

  if (!d.data.is_leaf) {

    const featureIndex = d.data.feature;
    let filteredData = treeData.data_feature.map(
      (row) => row[d.data.feature],
    );

    let combinedData = filteredData.map((value, index) => [
      value,
      treeData.data_target[index],
    ]);
    let indexSet = new Set();

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


    if (treeData.show_sample != "nodata" && treeData.show_sample != undefined) {
      stLog("debug", treeData.show_sample, "treedatasample");
      isSampleExist = true;
    }
    var isSampleExistInThisNode = true;
    if (isSampleExist) {
      isSampleExistInThisNode = true;
    }
    else {
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

    let indexesToRemove = Array.from(new Set([...indexSet]));

    indexesToRemove.sort((a, b) => b - a);

    indexesToRemove.forEach((index) => {
      if (index >= 0 && index < combinedData.length) {
        combinedData.splice(index, 1);
      }
    });

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

    let combinedDataValues = combinedData.map((item) => item[0]);

    function calculateAverages(data, threshold) {
      let sumBelowThreshold = 0;
      let countBelowThreshold = 0;
      let sumAboveOrEqualThreshold = 0;
      let countAboveOrEqualThreshold = 0;

      data.forEach((item) => {
        if (item[0] < threshold) {
          sumBelowThreshold += item[1];
          countBelowThreshold++;
        } else {
          sumAboveOrEqualThreshold += item[1];
          countAboveOrEqualThreshold++;
        }
      });

      const averageBelowThreshold =
        sumBelowThreshold / countBelowThreshold;
      const averageAboveOrEqualThreshold =
        sumAboveOrEqualThreshold / countAboveOrEqualThreshold;

      return {
        averageBelowThreshold,
        averageAboveOrEqualThreshold,
      };
    }

    const average = calculateAverages(
      combinedData,
      d.data.threshold,
    );
    xExtent = d3.extent(combinedDataValues)
    if (Number.isNaN(xExtent[0]) || xExtent[0] === undefined || xExtent[0] > d.data.threshold) {
      xExtent[0] = d.data.threshold - 0.2;
    }
    if (Number.isNaN(xExtent[1]) || xExtent[1] === undefined || xExtent[1] < d.data.threshold) {
      xExtent[1] = d.data.threshold + 0.2;
    }

    if(isSampleExistInThisNode){
    if (xExtent[0] > treeData.show_sample[featureIndex]) {
      xExtent[0] = treeData.show_sample[featureIndex] - 0.2
    }
    if (xExtent[1] < treeData.show_sample[featureIndex]) {
      xExtent[1] = treeData.show_sample[featureIndex] + 0.2
    }
    }


    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .nice()
      .range([0, scatterplotWidth]);

    const xDomain = xScale.domain();

    const xTickValues = [
      xDomain[0],
      d.data.threshold,
      xDomain[1],
    ];

    d3.select(this)
      .append("g")
      .attr(
        "transform",
        `translate(${-scatterplotWidth / 2}, ${scatterplotHeight})`,
      )
      .attr("class", "xAxis")
      .call(
        d3
          .axisBottom(xScale)
          .tickSize(0)
          .tickValues(xTickValues)
          .tickPadding(8),
      )
      .selectAll(".tick")
      .attr("class", "xAxis-text")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none")
      .style("fill", "black");

    d3.select(this).selectAll(".domain")
      .style("stroke", "black");
    yExtent = d3.extent(treeData.data_target)
    const yScale = d3
      .scaleLinear()
      .domain(yExtent)
      .nice()
      .range([scatterplotHeight, 0]);

      const yDomain = yScale.domain();

      const yTickValues = [
      yDomain[0],
      yDomain[0] + (yDomain[1] - yDomain[0]) / 2,
      yDomain[1],
    ];

    stLog("debug", yTickValues, "yTickValues");


    d3.select(this)
      .append("g")
      .call(
            d3
            .axisRight(yScale)
            .tickSize(0)
            .tickPadding(4)
            .tickValues(yTickValues)
            .tickFormat(d3.format(",.0f")),
      )
      .call((d) => d.select(".domain").remove())
      .attr("transform", `translate(${-scatterplotWidth / 2 - yAxisMargin}, 0)`)
      .attr("class", "yAxis")
      .selectAll(".tick")
      .attr("class", "yAxis-text")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none")
      .style("fill", "black");


    d3.select(this).selectAll(".domain")
      .style("stroke", "black");

    /*
                const GridLineH = function () {
                    return d3.axisLeft().scale(yScale);
                };
    
                d3.select(this)
                    .append("g")
                    .attr("class", "st-grid")
                    .call(
                        GridLineH()
                            .tickSize(-histogramWidth, 0, 0)
                            .tickFormat("")
                            .ticks(6),
                    )
                    .attr("transform", `translate(${-histogramWidth / 2}, ${0})`);
 
                const GridLineV = function () {
                    return d3.axisBottom().scale(xScale);
                };
 
                d3.select(this)
                    .append("g")
                    .attr("class", "st-grid")
                    .call(
                        GridLineV()
                            .tickSize(histogramHeight, 0, 0)
                            .tickFormat("")
                            .ticks(5),
                    )
                    .attr("transform", `translate(${-histogramWidth / 2}, ${0})`);
   */

    const mouseover = function(d) {
      tooltipBody.style("opacity", 1);
      d3.select(this).style("fill", "red");

      tooltipModal.style("opacity", 1);
      d3.select(this).style("fill", "red");
    };

    const mouseleave = function(d) {
      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");
      d3.select(this).style("fill", "#0099cc");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("fill", "#0099cc");
    };

    const mouseoverline = function(d) {
      tooltipBody.style("opacity", 1);
      d3.select(this).style("stroke", "red");

      tooltipModal.style("opacity", 1);
      d3.select(this).style("stroke", "red");
    };

    const mouseleaveline = function(d) {
      tooltipBody.style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("stroke", "black");

      tooltipModal.style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("stroke", "black");
    };

    const mousemovecircle = function(event, d) {
      tooltipBody
        .html(`<b>(X,Y):</b> (${parseFloat(d[0]).toFixed(3)}) (${parseFloat(d[1]).toFixed(3)})`)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(`<b>(X,Y):</b> (${parseFloat(d[0]).toFixed(3)}) (${parseFloat(d[1]).toFixed(3)})`)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    d3.select(this)
      .append("g")
      .selectAll("g")
      .data(combinedData)
      .join("circle")
      .attr("cx", (d, i) => xScale(combinedData[i][0]))
      .attr("cy", (d, i) => yScale(combinedData[i][1]))
      .attr("r", 2)
      .attr(
        "transform",
        `translate(${-scatterplotWidth / 2}, ${0})`,
      )
      .style("fill", "#0099cc")
      .style("fill-opacity", 0.5)
      .on("mouseover", mouseover)
      .on("mouseleave", mouseleave)
      .on("mousemove", mousemovecircle);

    const mousemoveavaragebelow = function(event, d) {
      tooltipBody
        .html(`<b>Average:</b> ${parseFloat(average.averageBelowThreshold).toFixed(3)} `)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(`<b>Average:</b> ${parseFloat(average.averageBelowThreshold).toFixed(3)} `)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    const mousemoveavarageabove = function(event, d) {
      tooltipBody
        .html(
          `<b>Average:</b> ${parseFloat(average.averageAboveOrEqualThreshold).toFixed(3)} `,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(
          `<b>Averge:</b> ${parseFloat(average.averageAboveOrEqualThreshold).toFixed(3)} `,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    const mousemoveAllData = function(event, d) {
      tooltipBody
        .html(
          `<b>All Data</b>
              <br>Impurity: ${d.data.impurity}
              <br>Samples: ${d.data.samples}
              <br>Threshold: ${parseFloat(d.data.threshold).toFixed(3)}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
        .style("opacity", 1);

      tooltipModal
        .html(
          `<b>All Data</b>:<br>
              <br>Impurity: ${d.data.impurity}
              <br>Samples: ${d.data.samples}
              <br>Threshold: ${parseFloat(d.data.threshold).toFixed(3)}`,
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px")
        .style("opacity", 1);
    };

    const mouseleaveAllData = function(event, d) {
      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");
    };

    var threshold = parseFloat(d.data.threshold).toFixed(3);
    d3.select(this)
      .append("line")
      .attr("class", "threshold-line")
      .attr("x1", xScale(threshold))
      .attr("x2", xScale(threshold))
      .attr("y1", 0)
      .attr("y2", scatterplotHeight)
      .attr("stroke", "black")

      .attr("transform", `translate(${-scatterplotWidth / 2},0)`)
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,5")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none");

    stLog("debug", average.averageBelowThreshold, "avarage 1:")
    if (!isNaN(average.averageBelowThreshold)) {
      d3.select(this)
        .append("line")
        .attr("class", "average-line")
        .attr("x1", 0)
        .attr("x2", xScale(threshold))
        .attr("y1", yScale(average.averageBelowThreshold))
        .attr("y2", yScale(average.averageBelowThreshold))
        .attr("stroke", "black")
        .attr("transform", `translate(${-histogramWidth / 2},0)`)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .on("mouseover", mouseoverline)
        .on("mouseleave", mouseleaveline)
        .on("mousemove", mousemoveavaragebelow)
        .style("user-select", "none")
        .style("-webkit-user-select", "none")
        .style("-moz-user-select", "none")
        .style("-ms-user-select", "none");
    }


    if (!isNaN(average.averageAboveOrEqualThreshold)) {
      d3.select(this)
        .append("line")
        .attr("class", "average-line")
        .attr("x1", xScale(threshold))
        .attr("x2", xScale(xDomain[1]))
        .attr("y1", yScale(average.averageAboveOrEqualThreshold))
        .attr("y2", yScale(average.averageAboveOrEqualThreshold))
        .attr("stroke", "black")
        .attr("transform", `translate(${-histogramWidth / 2},0)`)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .on("mouseover", mouseoverline)
        .on("mouseleave", mouseleaveline)
        .on("mousemove", mousemoveavarageabove);
    }

    d3.select(this)
      .append("text")
      .attr("class", "st-target")
      .attr("x", 0)
      .attr("y", rectHeight + 15)
      .style("text-anchor", "middle")
      .style("font-size", "18px")
      .text(treeData.feature_names[d.data.feature])
      .on("mousemove", mousemoveAllData)
      .on("mouseleave", mouseleaveAllData)
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none");

    var mouseovertriangle = function(d) {


      tooltipBody.style("opacity", 1);
      tooltipModal.style("opacity", 1);

      d3.select(this).style("fill", "#0099cc").style("stroke", "#0099cc");
    };

    var mouseleavetriangle = function(d) {


      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("fill", "green").style("stroke", "green");
    };


    var mousemovetriangle = function(event, d) {

      tooltipModal

        .html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : 'N/A';
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipBody
        .html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : 'N/A';
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    var color = "green";
    var triangleSize = 25;
    var verticalTransform = histogramHeight - Math.sqrt(triangleSize) + 15;  // Dostosuj wysokość transformacji

    var triangle = d3.symbol()
      .type(d3.symbolTriangle)
      .size(triangleSize);

    if (isSampleExistInThisNode) {
      stLog("debug", d, "Exist");
      d3.select(this).append("path")
        .attr("d", triangle)
        .attr("class", "st-triangle")
        .style("stroke-width", 1)
        .style("stroke-opacity", 1)
        .style("fill", color)
        .attr("transform", function(d) {
          return "translate(" + (-histogramWidth / 2 + xScale(treeData.show_sample[featureIndex])) + "," + verticalTransform + ")";
        })
        .on("mouseover", mouseovertriangle)
        .on("mouseleave", mouseleavetriangle)
        .on("mousemove", mousemovetriangle);
    }
  }

}

export function processRegressionLeaf(
  d,
  treeData,
  scatterplotLeafWidth,
  scatterplotLeafHeight,
  rectHeight,
  tooltipBody,
  tooltipModal,
  click,
  showpath
) {
  if (d.data.is_leaf) {
    const featureIndex = d.data.feature;
    var isSampleExist = false;


    if (treeData.show_sample != "nodata" && treeData.show_sample !== undefined) {
      isSampleExist = true;
    }
    var isSampleExistInThisNode = isSampleExist;
    let filteredData = treeData.data_feature.map(
      (row) => row[d.parent.data.feature],
    );

    let combinedData = filteredData.map((value, index) => [
      value,
      treeData.data_target[index],
    ]);
    let indexSet = new Set();

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

    let indexesToRemove = Array.from(new Set([...indexSet]));

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

    d3.select(this)
      .append("rect")
      .attr("class", "histogram-background")
      .attr("x", -(scatterplotLeafWidth / 2) - 10)
      .attr("y", -10)
      .attr("width", rectHeight + 20)
      .attr("height", rectHeight - 10)
      .attr("stroke-width", 1)
      .attr("stroke", "#545454")
      .attr("rx", 10)
      .attr("ry", 10)
      .style("fill", "#ffffff")
      .on("click", click)
      .on("click", function() {
        showpath(nodeToClick)
      });

    let combinedDataValues = combinedData.map((item) => item[0]);

    function calculateAverage(data) {
      const length = data.length;
      const sum = data.reduce((accumulator, currentValue) => {
        return accumulator + currentValue[1];
      }, 0);

      return sum / length;
    }

    const average = calculateAverage(combinedData);
    var xExtent = d3.extent(combinedDataValues);
    xExtent[0] = xExtent[0] - 0.2;
    xExtent[1] = xExtent[1] + 0.2;


    if(isSampleExistInThisNode){
    if (xExtent[0] > treeData.show_sample[featureIndex]) {
      xExtent[0] = treeData.show_sample[featureIndex] - 0.2
    }
    if (xExtent[1] < treeData.show_sample[featureIndex]) {
      xExtent[1] = treeData.show_sample[featureIndex] + 0.2
    }
  }

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .nice()
      .range([0, scatterplotLeafWidth]);
    //TODO makeXAxis
    // const xTickValues = [
    //   xDomain[0],
    //   xDomain[0]+(xDomain[1]-xDomain[0])/2,
    //   xDomain[1],
    // ];
    //
    //
    //
    // d3.select(this)
    //   .append("g")
    //   .attr(
    //     "transform",
    //     `translate(${-scatterplotLeafWidth / 2 + 15}, ${scatterplotLeafHeight})`,
    //   )
    //   .call(
    //     d3.axisBottom(xScale)
    //         .tickSize(0)
    //         .tickPadding(4)
    //         .tickValues(xTickValues)
    //         .tickFormat(d3.format(",.0f")),
    //   );
    


    yExtent = d3.extent(treeData.data_target)

    const yScale = d3
      .scaleLinear()
      .domain(d3.extent(yExtent))
      .nice()
      .range([scatterplotLeafHeight, 0]);


    const yDomain = yScale.domain();

      const yTickValues = [
      yDomain[0],
      yDomain[0] + (yDomain[1] - yDomain[0]) / 2,
      yDomain[1],
    ];

    stLog("debug", yTickValues, "yTickValues");


    d3.select(this)
      .append("g")
      .call(
            d3
            .axisRight(yScale)
            .tickSize(0)
            .tickPadding(4)
            .tickValues(yTickValues)
            .tickFormat(d3.format(",.0f")),
      )
      .call((d) => d.select(".domain").remove())
      .attr(
        "transform",
        `translate(${-scatterplotLeafWidth / 2 + 15 - yAxisMargin}, 0)`,
      )
      .attr("class", "yAxis")
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none")
      .style("fill", "black");


    d3.select(this).selectAll(".domain")
      .style("stroke", "black");
    /*
    const GridLineH = function () {
        return d3.axisLeft().scale(yScale);
    };
 
    d3.select(this)
        .append("g")
        .attr("class", "st.grid")
        .call(
            GridLineH()
                .tickSize(-histogramWidth, 0, 0)
                .tickFormat("")
                .ticks(6),
        )
        .attr("transform", `translate(${-histogramWidth / 2}, ${0})`);
 
    const GridLineV = function () {
        return d3.axisBottom().scale(xScale);
    };
    d3.select(this)
        .append("g")
        .attr("class", "st.grid")
        .call(
            GridLineV()
                .tickSize(histogramHeight, 0, 0)
                .tickFormat("")
                .ticks(5),
        )
        .attr("transform", `translate(${-histogramWidth / 2}, ${0})`);
      */

    const mouseover = function(d) {
      tooltipBody.style("opacity", 1);
      d3.select(this).style("fill", "red");

      tooltipModal.style("opacity", 1);
      d3.select(this).style("fill", "red");
    };

    const mouseleave = function(d) {
      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("fill", "#0099cc");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("fill", "#0099cc");
    };

    const mouseoverline = function(d) {
      tooltipBody.style("opacity", 1);
      d3.select(this).style("stroke", "red");

      tooltipModal.style("opacity", 1);
      d3.select(this).style("stroke", "red");
    };

    const mouseleaveline = function(d) {
      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("stroke", "black");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("stroke", "black");
    };

    const mousemovecircle = function(event, d) {
      tooltipBody
        .html(`<b>(X,Y):</b> (${parseFloat(d[0]).toFixed(3)}) (${parseFloat(d[1]).toFixed(3)})`)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(`<b>(X,Y):</b> (${parseFloat(d[0]).toFixed(3)}) (${parseFloat(d[1]).toFixed(3)})`)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    const mousemoveavarage = function(event, d) {
      tooltipBody
        .html(`<b>Average:</b> ${parseFloat(average).toFixed(3)} `)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipModal
        .html(`<b>Average:</b> ${parseFloat(average).toFixed(3)} `)
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };

    d3.select(this)
      .append("g")
      .selectAll("g")
      .data(combinedData)
      .join("circle")
      .attr("cx", (d, i) => xScale(combinedData[i][0]))
      .attr("cy", (d, i) => yScale(combinedData[i][1]))
      .attr("r", 2)
      .attr(
        "transform",
        `translate(${-scatterplotLeafWidth / 2 + 15}, ${0})`,
      )
      .style("fill", "#0099cc")
      .style("fill-opacity", 0.5)
      .on("mouseover", mouseover)
      .on("mouseleave", mouseleave)
      .on("mousemove", mousemovecircle);



    if (!isNaN(average)) {
      d3.select(this)
        .append("line")
        .attr("class", "average-line")
        .attr("x1", 0)
        .attr("x2", scatterplotLeafWidth)
        .attr("y1", yScale(average))
        .attr("y2", yScale(average))
        .attr("stroke", "black")
        .attr(
          "transform",
          `translate(${-scatterplotLeafWidth / 2 + 15},0)`,
        )
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .on("mouseover", mouseoverline)
        .on("mouseleave", mouseleaveline)
        .on("mousemove", mousemoveavarage);
    }

    function wrapText(text, width) {
      text.each(function() {
        var textElement = d3.select(this),
          words = textElement.text().split(/\s+/).reverse(),
          word,
          line = [],
          lineNumber = 0,
          lineHeight = 1.1,
          x = textElement.attr("x"),
          y = textElement.attr("y"),
          dy = 0,
          tspan = textElement.text(null).append("tspan").attr("x", x).attr("y", y);

        while (word = words.pop()) {
          line.push(word);
          tspan.text(line.join(" "));
          if (tspan.node().getComputedTextLength() > width && line.length > 1) {
            line.pop();
            tspan.text(line.join(" "));
            line = [word];
            tspan = textElement.append("tspan")
              .attr("class", "st-target")
              .attr("x", x)
              .attr("y", y)
              .attr("dy", ++lineNumber * lineHeight + "em")
              .text(word);
          }
        }
      });
    }

    var maxWidth = 100;

    var textElement = d3.select(this)
      .append("text")
      .attr("class", "st-target")
      .attr("x", 15)
      .attr("y", rectHeight + 5)
      .style("text-anchor", "middle")
      .style("font-size", "18px")
      .text(`${d.data.treeclass} = ${parseFloat(d.data.class_distribution[0][0].toFixed(3))}`)
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none")
      .call(wrapText, maxWidth);

    var lineCount = textElement.selectAll("tspan").size();
    var lineHeightEm = 1.1;
    var fontSizePx = 18;
    var textSpacing = 5;

    var nTextY = rectHeight + (lineCount * lineHeightEm * fontSizePx) + textSpacing;

    d3.select(this)
      .append("text")
      .attr("class", "st-target")
      .attr("x", 15)
      .attr("y", nTextY)
      .style("text-anchor", "middle")
      .style("font-size", "18px")
      .text(`n = ${d.data.samples}`)
      .style("user-select", "none")
      .style("-webkit-user-select", "none")
      .style("-moz-user-select", "none")
      .style("-ms-user-select", "none");




    var mouseovertriangle = function(d) {


      tooltipBody.style("opacity", 1);
      tooltipModal.style("opacity", 1);

      d3.select(this).style("fill", "#0099cc").style("stroke", "#0099cc");
    };

    var mouseleavetriangle = function(d) {


      tooltipBody
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      tooltipModal
        .style("opacity", 0)
        .style("top", -2000 + "px")
        .style("left", -2000 + "px");

      d3.select(this).style("fill", "green")
        .style("stroke", "green");
    };


    var mousemovetriangle = function(event, d) {

      tooltipModal
        .html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : 'N/A';
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");

      tooltipBody
        .html(
          treeData.feature_names.map((feature, index) => {
            const sampleValue = treeData.show_sample[index];
            const formattedValue = !isNaN(parseFloat(sampleValue)) ? parseFloat(sampleValue).toFixed(3) : 'N/A';
            return `<b>${feature}:</b> ${formattedValue}`;
          }).join(",")
        )
        .style("top", event.pageY - 10 + "px")
        .style("left", event.pageX + 10 + "px");
    };


    var color = "green";
    var triangleSize = 25;

    var triangle = d3.symbol()
      .type(d3.symbolTriangle)
      .size(triangleSize);
    stLog("debug", d, "XSCALE");
    if (isSampleExistInThisNode) {
      stLog("debug", d, "Existtt");
      d3.select(this).append("path")
        .attr("d", triangle)
        .attr("class", "st-triangle")
        .style("stroke", color)
        .style("stroke-width", 1)
        .style("stroke-opacity", 1)
        .style("fill", color)
        .attr("transform", function(d) {
          return "translate(" + (-scatterplotLeafWidth / 2 + 15 + xScale(treeData.show_sample[d.parent.data.feature])) + "," + scatterplotLeafHeight + ")";
        })
        .on("mouseover", mouseovertriangle)
        .on("mouseleave", mouseleavetriangle)
        .on("mousemove", mousemovetriangle);
    }
  }
}

export function ipytest(){
  stLog("debug","to działa","lolxd")
}
