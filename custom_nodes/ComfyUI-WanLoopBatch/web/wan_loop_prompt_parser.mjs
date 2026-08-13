const SLOT_COUNT = 10;

function stripOptionalNumber(prompt) {
  return prompt.replace(/^\s*\d{1,2}(?:[.):]|：)\s*/, "").trim();
}

export function parsePromptsText(rawText, filename = "prompts.txt") {
  const text = String(rawText)
    .replace(/^\uFEFF/, "")
    .replace(/\r\n?/g, "\n")
    .trim();
  if (!text) throw new Error(`${filename} is empty`);

  let prompts;
  if (/^\[/.test(text)) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      throw new Error(`${filename} contains invalid JSON: ${error.message}`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${filename} JSON must be an array`);
    prompts = parsed.map((item) => String(item).trim());
  } else if (/^\s*---+\s*$/m.test(text)) {
    // Explicit separator retained for backward compatibility.
    prompts = text
      .split(/^\s*---+\s*$/m)
      .map((item) => item.trim())
      .filter(Boolean);
  } else if (/\n[ \t]*\n/.test(text)) {
    // Preferred format: each prompt is a multiline block, and one or more
    // empty lines separate it from the prompt for the next image.
    prompts = text
      .split(/\n[ \t]*\n+/)
      .map((item) => item.trim())
      .filter(Boolean);
    if (prompts.length !== SLOT_COUNT) {
      throw new Error(
        `${filename} must contain exactly 10 blank-line-separated prompt blocks; found ${prompts.length}`,
      );
    }
  } else {
    // Old one-line-per-prompt files continue to work.
    prompts = text
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  prompts = prompts.map(stripOptionalNumber);
  if (prompts.length !== SLOT_COUNT) {
    throw new Error(`${filename} must contain exactly 10 prompts; found ${prompts.length}`);
  }
  if (prompts.some((item) => !item)) throw new Error(`${filename} contains an empty prompt`);
  return prompts;
}
