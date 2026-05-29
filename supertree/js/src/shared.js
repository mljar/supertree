export const DEBUG_LEVEL = "debug";
export const yAxisMargin = 25;

export function stLog(level, obj, message = "Data;") {
  const levels = ["debug", "custom", "info", "warning", "error"];

  if (levels.indexOf(level) >= levels.indexOf(DEBUG_LEVEL)) {
    console.log(`[${level.toUpperCase()}]:`, message, obj);
  }
}
