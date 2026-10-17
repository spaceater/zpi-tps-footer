/**
 * session-tps-footer — session id and live throughput in pi's footer.
 *
 * pi exposes no way to append to its built-in footer, so this extension
 * REPLACES it via ctx.ui.setFooter(). It reproduces pi's built-in footer and
 * adds exactly two elements:
 *   - line 1 right-aligned: the session id
 *   - line 2 after the model name: `• <t/s>`, always rendered, `NULL t/s`
 *     until the first measurement (no `~` estimate marker)
 *
 * Everything else matches that footer, so re-check it after every pi update.
 * Source of truth: `dist/modes/interactive/components/footer.js` in the pi
 * binary, verified against pi 1.2.0; extract it with
 *
 *     python3 -c 'd=open("/opt/pi/pi","rb").read(); \
 *       s=d.find(b"// dist/modes/interactive/components/footer.js"); \
 *       print(d[s:d.find(b"// dist/", s+10)].decode())'
 *
 * What to compare (pi 1.2.0):
 *   - cwd comes from sessionManager.getCwd() (live), shortened via
 *     path.resolve/path.relative
 *   - usage totals include `entry.type === "usage"`; addUsageToTotals reads
 *     input/output/cacheRead/cacheWrite/cost.total without defaults
 *   - `(sub)`: `kimi-coding` or modelRuntime.isUsingSubscription(provider)
 *     (= isUsingOAuth && provider auth.oauth.isSubscription). ModelRegistry
 *     does not expose isUsingSubscription, so the same composite is spelled
 *     out here
 *   - `• xp` badge: process.env.PI_EXPERIMENTAL === "1"
 *   - `(auto)` hint: pi.getSettings().compaction?.enabled ?? true
 *   - routed display `→ <physical model> • <level>` for virtual models
 *   - context percent is "?" only when contextUsage.percent === null; since
 *     1.2.0 the session measures limits against the routed physical model
 *     (session._limitsModel), which ctx.getContextUsage() picks up here too
 *   - stats are cached like the built-in getSessionStats(): recomputed only
 *     when session, sessionId, leafId, entryCount or the model changes
 *   - status line: sorted by key, joined with " ", empty entries not filtered
 *   - `• <session name>` suffix on line 1 when a session name is set
 *
 * TpsMeter and its calibration come from the pi-tps-live package.
 */

import type { ExtensionAPI, ExtensionContext, ReadonlyFooterDataProvider } from "@earendil-works/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import { isAbsolute, relative, resolve, sep } from "path";

// --- measurement ------------------------------------------------------------

/**
 * Tokens-per-second tracking, no pi imports.
 *
 * Measurement model:
 * - Decode time starts at the first streamed delta, not at message start.
 *   Including time-to-first-token would let a long thinking phase dominate
 *   the denominator and report meaningless low numbers.
 * - Providers that stream cumulative usage (e.g. Anthropic `output_tokens`)
 *   give exact token counts during the stream.
 * - Providers that only report usage in the final chunk (most
 *   OpenAI-compatible APIs) are estimated from characters. The estimate uses
 *   an ASCII-equivalent character count divided by a chars-per-token ratio
 *   calibrated from completed turns (shrunk toward a prior), so it adapts to
 *   prose vs. code vs. CJK content.
 */

/** A tok/s reading, either live (during a stream) or final (after one). */
interface TpsReading {
	/** Tokens per second over the measured interval. */
	tps: number;
	/** Tokens counted over the measured interval. */
	tokens: number;
	/** Wall-clock seconds of the measured interval. */
	seconds: number;
	/** True when tokens were estimated from characters (provider streamed no usage). */
	estimated: boolean;
}

/** Live sliding window length, in ms. */
const LIVE_WINDOW_MS = 5000;
/** Minimum decode time before a live reading is reported. */
const MIN_LIVE_MS = 600;
/** Minimum token count before a live reading is reported. */
const MIN_LIVE_TOKENS = 8;
/** Minimum sliding-window span; shorter windows fall back to the whole-stream rate. */
const MIN_WINDOW_MS = 800;

// --- chars-per-token calibration -------------------------------------------
// The live estimate divides ASCII-equivalent characters by this ratio. It is
// learned from completed turns and shrunk toward a code-leaning prior so the
// first turns are still reasonable. At finish time the authoritative
// `usage.output` replaces the estimate, so the prior never pollutes results.
const PRIOR_CHARS_PER_TOKEN = 4;
const PRIOR_WEIGHT_TOKENS = 500;
const MIN_CHARS_PER_TOKEN = 1.2;
const MAX_CHARS_PER_TOKEN = 6;
/** Non-ASCII (CJK / full-width) characters carry ~2.5x the token density of ASCII. */
const NON_ASCII_WEIGHT = 2.5;

/**
 * Count characters in "ASCII-equivalent" units: ASCII counts 1, non-ASCII
 * counts 2.5. Dividing the result by ~4 approximates a token count for mixed
 * prose/code/CJK text (CJK runs ~1.6 chars per token, ASCII ~4).
 */
function asciiEquivalentChars(text: string): number {
	let ascii = 0;
	let wide = 0;
	for (const ch of text) {
		if (ch.codePointAt(0)! < 128) ascii++;
		else wide++;
	}
	return ascii + wide * NON_ASCII_WEIGHT;
}

interface LiveSample {
	/** Timestamp (ms, same clock as the meter's callers). */
	t: number;
	/** Cumulative estimated tokens at that time (float). */
	tokens: number;
}

/**
 * Per-assistant-message stream meter.
 *
 * Usage: `start()` on message_start, `addDelta(text, now, usageOutput?)`
 * for every text/thinking/toolcall delta, `live(now)` for the footer, and
 * `finish(now, { outputTokens, ok })` on message_end.
 */
class TpsMeter {
	private calEquiv = 0;
	private calTokens = 0;

	private streaming = false;
	/** Timestamp of the first streamed delta; 0 until one arrives. */
	private decodeStartMs = 0;
	private equiv = 0;
	/** Largest cumulative provider-reported output token count seen in this stream. */
	private providerTokens = 0;
	private samples: LiveSample[] = [];
	private lastReading: TpsReading | undefined;

	/** Begin tracking a new assistant stream. */
	start(): void {
		this.streaming = true;
		this.decodeStartMs = 0;
		this.equiv = 0;
		this.providerTokens = 0;
		this.samples = [];
	}

	/**
	 * Feed one streamed delta.
	 *
	 * @param text delta content (text, thinking or tool-call JSON)
	 * @param now timestamp
	 * @param providerOutputTokens cumulative provider-reported output tokens, when streamed
	 */
	addDelta(text: string, now: number, providerOutputTokens?: number): void {
		// Defensive: if message_start was missed, begin now so data is not lost.
		if (!this.streaming) this.start();
		if (this.decodeStartMs === 0) this.decodeStartMs = now;

		this.equiv += asciiEquivalentChars(text);
		if (providerOutputTokens !== undefined && providerOutputTokens > this.providerTokens) {
			this.providerTokens = providerOutputTokens;
		}

		this.samples.push({ t: now, tokens: this.estimatedTokens() });
		// Drop samples that fell out of the sliding window, keeping at least one.
		const cutoff = now - LIVE_WINDOW_MS;
		let drop = 0;
		while (drop < this.samples.length - 1 && this.samples[drop].t < cutoff) drop++;
		if (drop > 0) this.samples.splice(0, drop);
	}

	/** Estimated tokens accumulated in the current stream (float). */
	estimatedTokens(): number {
		return this.equiv / this.charsPerToken();
	}

	/** Current calibrated ASCII-equivalent chars per token. */
	charsPerToken(): number {
		const ratio =
			(this.calEquiv + PRIOR_CHARS_PER_TOKEN * PRIOR_WEIGHT_TOKENS) / (this.calTokens + PRIOR_WEIGHT_TOKENS);
		return Math.min(MAX_CHARS_PER_TOKEN, Math.max(MIN_CHARS_PER_TOKEN, ratio));
	}

	/**
	 * Live reading for the current stream, or undefined while there is not
	 * enough signal yet. Never throws; safe to call every render.
	 */
	live(now: number): TpsReading | undefined {
		if (!this.streaming || this.decodeStartMs === 0) return undefined;
		const seconds = (now - this.decodeStartMs) / 1000;
		const exact = this.providerTokens > 0;
		const tokens = exact ? this.providerTokens : this.estimatedTokens();
		if (seconds <= 0 || seconds * 1000 < MIN_LIVE_MS || tokens < MIN_LIVE_TOKENS) return undefined;

		if (exact) {
			return { tps: tokens / seconds, tokens, seconds, estimated: false };
		}

		// Estimated path: prefer a sliding-window rate for a "live" feel; fall
		// back to the whole-stream average when the window is too short.
		const first = this.samples[0];
		if (first && this.samples.length >= 2) {
			const spanMs = now - first.t;
			const windowTokens = tokens - first.tokens;
			if (spanMs >= MIN_WINDOW_MS && windowTokens > 0) {
				return { tps: windowTokens / (spanMs / 1000), tokens, seconds, estimated: true };
			}
		}
		return { tps: tokens / seconds, tokens, seconds, estimated: true };
	}

	/**
	 * Finish the current stream and remember the reading for idle display.
	 *
	 * @param now timestamp
	 * @param opts.outputTokens final `usage.output` for this message, when available
	 * @param opts.ok true when the turn completed cleanly (calibration uses the
	 *   real token count only then; aborted/errored streams are still measured
	 *   but not learned from)
	 */
	finish(now: number, opts?: { outputTokens?: number; ok?: boolean }): TpsReading | undefined {
		if (!this.streaming) return this.lastReading;
		this.streaming = false;

		// Nothing was streamed (immediate error): keep the previous reading.
		if (this.decodeStartMs === 0) return this.lastReading;

		const seconds = (now - this.decodeStartMs) / 1000;
		const actual = Math.max(0, Math.round(opts?.outputTokens ?? 0));
		const ok = opts?.ok ?? true;
		const tokens = actual > 0 ? actual : Math.round(this.estimatedTokens());

		if (actual > 0 && this.equiv > 0 && ok) {
			this.calEquiv += this.equiv;
			this.calTokens += actual;
		}
		if (seconds < 0.05 || tokens <= 0) {
			this.lastReading = undefined;
			return undefined;
		}
		this.lastReading = { tps: tokens / seconds, tokens, seconds, estimated: actual === 0 };
		return this.lastReading;
	}

	/** Last completed reading (shown while idle), if any. */
	get last(): TpsReading | undefined {
		return this.lastReading;
	}

	/** Whether a stream is currently being measured. */
	isStreaming(): boolean {
		return this.streaming;
	}

	/** Clear stream state and the idle reading, keeping calibration. */
	reset(): void {
		this.streaming = false;
		this.decodeStartMs = 0;
		this.equiv = 0;
		this.providerTokens = 0;
		this.samples = [];
		this.lastReading = undefined;
	}

	/** Clear learned calibration, restoring the prior. */
	resetCalibration(): void {
		this.calEquiv = 0;
		this.calTokens = 0;
	}
}

// --- footer rendering -------------------------------------------------------

/**
 * Reproduces pi's built-in footer layout (pi 1.2.0
 * `dist/modes/interactive/components/footer.js`) and adds two elements: the
 * session id right-aligned at the end of the **first** line, and a t/s readout
 * after the model name on the **second**.
 */

/** Minimal theme surface used for coloring. */
interface ThemeLike {
	fg(color: string, text: string): string;
	bold(text: string): string;
}

/** Minimal pi-tui surface used for layout. */
interface TuiLike {
	visibleWidth(text: string): number;
	truncateToWidth(text: string, width: number, ellipsis?: string): string;
}

/** Speed text + whether it describes live streaming. */
interface SpeedDisplay {
	text: string;
	live: boolean;
}

interface UsageTotals {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cost: number;
}

interface UsageLike {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cost: { total: number };
}

interface SessionEntryLike {
	type: string;
	message?: {
		role?: string;
		provider?: string;
		model?: string;
		thinkingLevel?: string;
		stopReason?: string;
		usage?: UsageLike;
	};
	usage?: UsageLike;
}

/** Minimum columns reserved for the left side of line 1 before the session id is dropped. */
const MIN_PWD_WIDTH = 12;

/** Format token counts for the compact footer (same buckets as pi's built-in footer). */
function formatTokens(count: number): string {
	if (count < 1000) return count.toString();
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
	return `${Math.round(count / 1000000)}M`;
}

/** Replace the home prefix with `~` when the cwd is inside it (pi semantics; the cwd comes from sessionManager.getCwd(), live since 1.2.0). */
function formatCwdForFooter(cwd: string, home: string | undefined): string {
	if (!home) return cwd;
	const resolvedCwd = resolve(cwd);
	const resolvedHome = resolve(home);
	const relativeToHome = relative(resolvedHome, resolvedCwd);
	const isInsideHome =
		relativeToHome === "" ||
		(relativeToHome !== ".." && !relativeToHome.startsWith(`..${sep}`) && !isAbsolute(relativeToHome));
	if (!isInsideHome) return cwd;
	return relativeToHome === "" ? "~" : `~${sep}${relativeToHome}`;
}

/** Collapse control characters so extension statuses stay single-line. */
function sanitizeStatusText(text: string): string {
	return text
		.replace(/[\r\n\t]/g, " ")
		.replace(/ +/g, " ")
		.trim();
}

/** Format a reading for display, e.g. `42 t/s` — whole tokens per second, rounded. */
function formatSpeedText(reading: { tps: number }): string {
	return `${Math.round(reading.tps)} t/s`;
}

/** Shown on line 2 until the first measurement of a session exists. */
const SPEED_PLACEHOLDER = "NULL t/s";

/**
 * Sum session usage the same way the built-in footer does: usage entries,
 * assistant messages, tool-result usage and branch/compaction summary usage.
 */
function collectUsageTotals(entries: Iterable<SessionEntryLike>): {
	totals: UsageTotals;
	latestCacheHitRate: number | undefined;
} {
	const totals: UsageTotals = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 };
	let latestCacheHitRate: number | undefined;
	const addUsage = (usage: UsageLike): void => {
		totals.input += usage.input;
		totals.output += usage.output;
		totals.cacheRead += usage.cacheRead;
		totals.cacheWrite += usage.cacheWrite;
		totals.cost += usage.cost.total;
	};
	for (const entry of entries) {
		if (entry.type === "usage" && entry.usage) {
			addUsage(entry.usage);
		} else if (entry.type === "message" && entry.message?.role === "assistant" && entry.message.usage) {
			addUsage(entry.message.usage);
			const latestPromptTokens =
				entry.message.usage.input + entry.message.usage.cacheRead + entry.message.usage.cacheWrite;
			latestCacheHitRate =
				latestPromptTokens > 0 ? (entry.message.usage.cacheRead / latestPromptTokens) * 100 : undefined;
		} else if (entry.type === "message" && entry.message?.role === "toolResult" && entry.message.usage) {
			addUsage(entry.message.usage);
		} else if ((entry.type === "branch_summary" || entry.type === "compaction") && entry.usage) {
			addUsage(entry.usage);
		}
	}
	return { totals, latestCacheHitRate };
}

// --- routed models ----------------------------------------------------------

/** The routed physical model behind a virtual model, when the built-in footer shows one. */
interface RoutedModel {
	modelId: string;
	thinkingLevel: string | undefined;
}

const VIRTUAL_MODEL_API = "pi-virtual";

/** Mirror of pi's routed-model lookup (session.routedModel): only virtual models
 * route, and only to a physical model. The model's own provider field is never
 * read — only its api tag. */
function resolveRoutedModel(
	model: { api?: string } | undefined,
	entries: Iterable<SessionEntryLike>,
	registry: { find(provider: string, modelId: string): { id: string; api?: string } | undefined } | undefined,
): RoutedModel | undefined {
	if (!model || model.api !== VIRTUAL_MODEL_API || !registry) return undefined;
	let latest: SessionEntryLike["message"];
	for (const entry of entries) {
		if (entry.type !== "message" || entry.message?.role !== "assistant") continue;
		if (entry.message.stopReason === "error" || entry.message.stopReason === "aborted") continue;
		latest = entry.message;
	}
	if (!latest?.provider || !latest.model) return undefined;
	const physical = registry.find(latest.provider, latest.model);
	if (!physical || physical.api === VIRTUAL_MODEL_API) return undefined;
	return { modelId: physical.id, thinkingLevel: latest.thinkingLevel };
}

/** Everything `renderFooterLines` needs, already resolved by the wiring below. */
interface FooterData {
	cwd: string;
	home: string | undefined;
	branch: string | null;
	sessionName: string | undefined;
	totals: UsageTotals;
	latestCacheHitRate: number | undefined;
	contextPercent: number | null | undefined;
	contextWindow: number;
	autoCompactEnabled: boolean;
	modelId: string | undefined;
	modelProvider: string | undefined;
	modelReasoning: boolean;
	thinkingLevel: string | undefined;
	providerCount: number;
	usingSubscription: boolean;
	statuses: ReadonlyMap<string, string>;
	speed: SpeedDisplay;
	sessionId: string;
	routedModel: RoutedModel | undefined;
	experimentalFeatures: boolean;
}

/**
 * Build the footer lines.
 *
 * Line 1: `pwd (branch) • session` on the left, session id right-aligned.
 * Line 2: token/cache/cost/context stats on the left, `model • t/s` on the right.
 * Line 3+: extension statuses (from `ctx.ui.setStatus`).
 */
function renderFooterLines(data: FooterData, width: number, theme: ThemeLike, tui: TuiLike): string[] {
	// --- first line: pwd + session name, with the session id on the right ---
	let pwd = formatCwdForFooter(data.cwd, data.home);
	if (data.branch) pwd = `${pwd} (${data.branch})`;
	if (data.sessionName) pwd = `${pwd} • ${data.sessionName}`;
	const pwdText = theme.fg("dim", pwd);

	let line1 = tui.truncateToWidth(pwdText, width, theme.fg("dim", "..."));
	if (data.sessionId) {
		const idText = theme.fg("dim", data.sessionId);
		const idWidth = tui.visibleWidth(idText);
		if (MIN_PWD_WIDTH + idWidth <= width) {
			const left = tui.truncateToWidth(pwdText, width - idWidth - 1, theme.fg("dim", "..."));
			const gap = Math.max(1, width - tui.visibleWidth(left) - idWidth);
			line1 = left + " ".repeat(gap) + idText;
		}
	}

	// --- second line: stats left, model + speed right ---
	const statsParts: string[] = [];
	if (data.totals.input) statsParts.push(`↑${formatTokens(data.totals.input)}`);
	if (data.totals.output) statsParts.push(`↓${formatTokens(data.totals.output)}`);
	if (data.totals.cacheRead) statsParts.push(`R${formatTokens(data.totals.cacheRead)}`);
	if (data.totals.cacheWrite) statsParts.push(`W${formatTokens(data.totals.cacheWrite)}`);
	if ((data.totals.cacheRead > 0 || data.totals.cacheWrite > 0) && data.latestCacheHitRate !== undefined) {
		statsParts.push(`CH${data.latestCacheHitRate.toFixed(1)}%`);
	}
	if (data.totals.cost || data.usingSubscription) {
		statsParts.push(`$${data.totals.cost.toFixed(3)}${data.usingSubscription ? " (sub)" : ""}`);
	}

	const contextPercentValue = data.contextPercent ?? 0;
	const autoIndicator = data.autoCompactEnabled ? " (auto)" : "";
	const contextPercent = data.contextPercent !== null ? contextPercentValue.toFixed(1) : "?";
	const contextDisplay =
		contextPercent === "?"
			? `?/${formatTokens(data.contextWindow)}${autoIndicator}`
			: `${contextPercent}%/${formatTokens(data.contextWindow)}${autoIndicator}`;
	const contextStr =
		contextPercentValue > 90
			? theme.fg("error", contextDisplay)
			: contextPercentValue > 70
				? theme.fg("warning", contextDisplay)
				: contextDisplay;
	statsParts.push(contextStr);
	if (data.experimentalFeatures) {
		statsParts.push(`${theme.fg("dim", "•")} ${theme.bold(theme.fg("warning", "xp"))}`);
	}

	let statsLeft = statsParts.join(" ");
	let statsLeftWidth = tui.visibleWidth(statsLeft);
	if (statsLeftWidth > width) {
		statsLeft = tui.truncateToWidth(statsLeft, width, "...");
		statsLeftWidth = tui.visibleWidth(statsLeft);
	}

	let modelSide = data.modelId || "no-model";
	if (data.modelReasoning) {
		const level = data.thinkingLevel || "off";
		modelSide = level === "off" ? `${modelSide} • thinking off` : `${modelSide} • ${level}`;
	}
	if (data.routedModel) {
		const level = data.routedModel.thinkingLevel ? ` • ${data.routedModel.thinkingLevel}` : "";
		modelSide += ` → ${data.routedModel.modelId}${level}`;
	}
	if (data.providerCount > 1 && data.modelProvider) {
		const withProvider = `(${data.modelProvider}) ${modelSide}`;
		if (statsLeftWidth + 2 + tui.visibleWidth(withProvider) <= width) modelSide = withProvider;
	}
	// The tok/s readout follows the model name and is colored outside the dim
	// wrapper below, so its live accent color survives; the bullet stays dim.
	const speedSeg = `• ${data.speed.text}`;
	const speedText = `${theme.fg("dim", "•")} ${theme.fg(data.speed.live ? "accent" : "dim", data.speed.text)}`;
	const speedWidth = 3 + tui.visibleWidth(data.speed.text);

	const rightWidth = tui.visibleWidth(modelSide) + speedWidth;
	const minPadding = 2;
	let line2: string;
	if (statsLeftWidth + minPadding + rightWidth <= width) {
		line2 =
			theme.fg("dim", statsLeft) +
			" ".repeat(width - statsLeftWidth - rightWidth) +
			theme.fg("dim", modelSide) +
			` ${speedText}`;
	} else {
		const availableForRight = width - statsLeftWidth - minPadding;
		if (availableForRight > 0) {
			const truncatedRight = tui.truncateToWidth(`${modelSide} ${speedSeg}`, availableForRight, "");
			const truncatedWidth = tui.visibleWidth(truncatedRight);
			line2 =
				theme.fg("dim", statsLeft) +
				" ".repeat(Math.max(0, width - statsLeftWidth - truncatedWidth)) +
				theme.fg("dim", truncatedRight);
		} else {
			line2 = theme.fg("dim", statsLeft);
		}
	}

	// Dim both halves separately: statsLeft may contain color codes (context
	// percentage) that end with a reset, which would clear an outer dim wrapper.
	const lines = [line1, line2];

	const statuses = Array.from(data.statuses.entries())
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([, text]) => sanitizeStatusText(text));
	if (statuses.length > 0) {
		lines.push(tui.truncateToWidth(statuses.join(" "), width, theme.fg("dim", "...")));
	}
	return lines;
}

// --- extension wiring -------------------------------------------------------

/** Minimum ms between render requests while streaming. */
const RENDER_THROTTLE_MS = 150;

interface AssistantMessageLike {
	role: "assistant";
	usage?: { output?: number };
	stopReason?: string;
}

function isAssistant(message: { role?: string } | undefined): message is AssistantMessageLike {
	return message?.role === "assistant";
}

export default function sessionTpsFooterExtension(pi: ExtensionAPI): void {
	const meter = new TpsMeter();

	let footerInstalled = false;
	let renderTui: { requestRender(): void } | undefined;
	let lastRenderAt = Number.NEGATIVE_INFINITY;

	/** Speed text for the current moment: live when streaming, last reading when idle, placeholder when unknown. */
	function speedDisplay(): SpeedDisplay {
		if (meter.isStreaming()) {
			const live = meter.live(Date.now());
			if (live) return { text: formatSpeedText(live), live: true };
		}
		const last = meter.last;
		return last ? { text: formatSpeedText(last), live: false } : { text: SPEED_PLACEHOLDER, live: false };
	}

	/** Request a footer repaint, throttled unless forced. */
	function requestRender(force = false): void {
		if (!renderTui) return;
		const t = Date.now();
		if (!force && t - lastRenderAt < RENDER_THROTTLE_MS) return;
		lastRenderAt = t;
		renderTui.requestRender();
	}

	/** Cached entry-scan results, mirroring the built-in footer's
	 * getSessionStats() (pi 1.2.0): totals, cache-hit rate, context usage and
	 * the routed model only change when the session, the leaf, the entry count
	 * or the model changes, so a render does not rescan every session entry.
	 * ctx is a live view over the runner, so the cache survives session
	 * switches and still keys on the active session. */
	let cachedStats:
		| {
				key: string;
				totals: UsageTotals;
				latestCacheHitRate: number | undefined;
				contextPercent: number | null | undefined;
				contextWindow: number;
				routedModel: RoutedModel | undefined;
		  }
		| undefined;

	function sessionStats(
		ctx: ExtensionContext,
		model: { provider?: string; id?: string; contextWindow?: number; api?: string } | undefined,
	) {
		const sessionManager = ctx.sessionManager;
		const key = `${sessionManager.getSessionId()}|${sessionManager.getLeafId()}|${sessionManager.getEntryCount()}|${model ? `${model.provider}/${model.id}` : ""}`;
		if (cachedStats && cachedStats.key === key) return cachedStats;
		const entries = ctx.sessionManager.getEntries();
		const { totals, latestCacheHitRate } = collectUsageTotals(entries);
		const contextUsage = ctx.getContextUsage();
		cachedStats = {
			key,
			totals,
			latestCacheHitRate,
			contextPercent: contextUsage?.percent,
			contextWindow: contextUsage?.contextWindow ?? model?.contextWindow ?? 0,
			routedModel: resolveRoutedModel(model, entries, ctx.modelRegistry),
		};
		return cachedStats;
	}

	function buildFooterData(ctx: ExtensionContext, footerData: ReadonlyFooterDataProvider): FooterData {
		const model = ctx.model;
		const stats = sessionStats(ctx, model);
		return {
			cwd: ctx.sessionManager.getCwd(),
			home: process.env.HOME || process.env.USERPROFILE,
			branch: footerData.getGitBranch(),
			sessionName: ctx.sessionManager.getSessionName(),
			totals: stats.totals,
			latestCacheHitRate: stats.latestCacheHitRate,
			contextPercent: stats.contextPercent,
			contextWindow: stats.contextWindow,
			autoCompactEnabled: pi.getSettings().compaction?.enabled ?? true,
			modelId: model?.id,
			modelProvider: model?.provider,
			modelReasoning: model?.reasoning ?? false,
			thinkingLevel: ctx.thinkingLevel,
			providerCount: footerData.getAvailableProviderCount(),
			// Subscription-backed providers pi ships a built-in hint for, plus
			// OAuth providers whose OAuth config is marked as a subscription.
			// Equivalent to the built-in footer's modelRuntime.isUsingSubscription()
			// (pi 1.2.0), which ModelRegistry does not expose.
			usingSubscription: model
				? model.provider === "kimi-coding" ||
					(ctx.modelRegistry.isUsingOAuth(model) &&
						ctx.modelRegistry.getProvider(model.provider)?.auth.oauth?.isSubscription === true)
				: false,
			statuses: footerData.getExtensionStatuses(),
			speed: speedDisplay(),
			sessionId: ctx.sessionManager.getSessionId() ?? "",
			routedModel: stats.routedModel,
			experimentalFeatures: process.env.PI_EXPERIMENTAL === "1",
		};
	}

	/** Install the custom footer (TUI mode only). Idempotent. */
	function installFooter(ctx: ExtensionContext): void {
		if (footerInstalled || ctx.mode !== "tui") return;
		footerInstalled = true;
		ctx.ui.setFooter((tui, theme, footerData) => {
			renderTui = tui;
			const unsubscribe = footerData.onBranchChange(() => tui.requestRender());
			return {
				dispose() {
					unsubscribe();
					if (renderTui === tui) renderTui = undefined;
				},
				invalidate() {},
				render(width: number) {
					return renderFooterLines(buildFooterData(ctx, footerData), width, theme, {
						visibleWidth,
						truncateToWidth,
					});
				},
			};
		});
	}

	// --- events ---------------------------------------------------------------

	pi.on("session_start", (_event, ctx) => {
		meter.reset();
		installFooter(ctx);
	});

	pi.on("model_select", () => {
		// Different model, different throughput: drop the stale idle reading
		// (calibration stays, it is content- and hardware-bound, not model-bound).
		meter.reset();
		requestRender(true);
	});

	pi.on("session_info_changed", () => {
		requestRender(true);
	});

	pi.on("message_start", (event, ctx) => {
		installFooter(ctx);
		if (isAssistant(event.message)) meter.start();
	});

	pi.on("message_update", (event, ctx) => {
		if (!isAssistant(event.message)) return;
		installFooter(ctx);
		const streamEvent = event.assistantMessageEvent;
		if (
			streamEvent.type === "text_delta" ||
			streamEvent.type === "thinking_delta" ||
			streamEvent.type === "toolcall_delta"
		) {
			meter.addDelta(streamEvent.delta, Date.now(), event.message.usage?.output);
		}
		requestRender();
	});

	pi.on("message_end", (event, _ctx) => {
		if (!isAssistant(event.message)) return;
		const { stopReason, usage } = event.message;
		const ok = stopReason === "stop" || stopReason === "length" || stopReason === "toolUse";
		meter.finish(Date.now(), { outputTokens: usage?.output, ok });
		requestRender(true);
	});
}
