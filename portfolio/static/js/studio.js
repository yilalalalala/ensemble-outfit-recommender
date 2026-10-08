// DS Studio: offline evaluation (/api/metrics), serving status (/readyz, /api/v2/meta, /api/v2/metrics),
// recommendation inspection (/api/home + /api/explain) and the Label workflow (/api/label/*).
// Values and labels are shown as the API returns them; formatting matches the pre-redesign DS view.
import {$, $$, DEMO, api, esc, getHome, probe, profileDetail, profileName, reasonList, shapBars, state} from "./core.js";

const pct = x => x == null || isNaN(x) ? "—" : (x * 100).toFixed(2) + "%";
const f5 = x => x == null || isNaN(x) ? "—" : x.toFixed(5);
const lift = x => x == null || isNaN(x) ? "—" : `<span class="${x > 0 ? "pos" : ""}">${x > 0 ? "+" : ""}${(x * 100).toFixed(1)}%</span>`;
const raw = x => x == null ? "" : `title="${esc(String(x))}"`;   // exact value on hover / for assistive tech
const num = (v, fmt) => `<td class="num" ${raw(v)}>${fmt(v)}</td>`;

function head(title, sub, tags = "") {
  return `<div class="studio-head"><div><p class="eyebrow">DS Studio</p><h1 tabindex="-1">${title}</h1>
    <p class="meta" style="margin-top:.4rem;max-width:60rem">${sub}</p></div><div>${tags}</div></div>`;
}
const table = (label, headHtml, rows, caption = "") =>
  `<div class="table-wrap" role="region" aria-label="${esc(label)}" tabindex="0"><table class="data">
    ${caption ? `<caption>${caption}</caption>` : ""}<thead><tr>${headHtml}</tr></thead><tbody>${rows}</tbody></table></div>`;
const th = (cols) => cols.map(c => typeof c === "string" ? `<th scope="col">${c}</th>` : `<th scope="col" class="num">${c[0]}</th>`).join("");
const loading = `<div class="state" role="status"><p class="state__title">Loading…</p></div>`;

// ---- Offline evaluation ---------------------------------------------------------------------------------
export async function evaluation(ctx) {
  ctx.setTitle("DS Studio · Evaluation");
  ctx.render(head("Offline evaluation", "Loading stored evaluation reports…") + loading);
  const m = await api("/api/metrics");
  if (!ctx.current()) return;
  const bv = m.m1_baselines_val || {}, bt = m.m1_baselines_test || {}, rv = m.m3_ranker_val?.metrics, rt = m.m3_ranker_test?.metrics;
  const best = "repeat_purchase+popularity_by_age";
  const liftA = (r, b) => r && b?.[best] ? lift((r["map@12"] - b[best]["map@12"]) / b[best]["map@12"]) : "—";
  const taRows = [...Object.entries(bv).map(([k, v]) => [k, "val", v]), rv && ["LightGBM LambdaRank", "val", rv],
                  ...Object.entries(bt).map(([k, v]) => [k, "test", v]), rt && ["LightGBM LambdaRank", "test", rt]].filter(Boolean)
    .map(([n, w, v]) => `<tr><th scope="row">${esc(n)}</th><td>${w}</td>${num(v["map@12"], f5)}${num(v["map@12_returning"], f5)}${num(v["map@12_new_customers"], f5)}${num(v["recall@12_cold_items"], pct)}</tr>`).join("");
  const rec = m.m2_retrieval_val || {};
  const recRows = Object.entries(rec).filter(([, v]) => v && v.recall != null)
    .map(([k, v]) => `<tr><th scope="row">${esc(k)}</th>${num(v.recall, pct)}<td class="num" ${raw(v.candidates_per_customer)}>${v.candidates_per_customer.toFixed(1)}</td>${num(rec.unique_to_channel?.[k], pct)}</tr>`).join("");
  const tbWeek = m.m4_track_b_test ? "test" : "validation";
  const tb = m.m4_track_b_test || m.m4_track_b_val;
  const tbRows = tb ? Object.entries(tb.results).map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td class="num" ${raw(v["relative_lift_recall@12_vs_popularity"])}>${lift(v["relative_lift_recall@12_vs_popularity"])}</td>${num(v["recall@12"], pct)}${num(v["ndcg@12"], f5)}${num(v["recall@12_tail"], pct)}${num(v["recall@12_cold"], pct)}${num(v["recall@12_jewellery"], pct)}${num(v["catalog_coverage@12"], pct)}</tr>`).join("") : "";
  const funnel = tb ? Object.entries(tb.funnel).map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td class="num">${Number(v).toLocaleString()}</td></tr>`).join("") : "";
  const empty = !taRows && !recRows && !tbRows;

  const root = ctx.render(head("Offline evaluation",
      "All numbers are offline (historical data). A launch decision would need an online A/B test."
      + (DEMO ? " This public build reads the same stored evaluation reports as the local application, published unchanged." : ""),
      `<span class="tag">Offline</span> <span class="tag">Stored reports: ${Object.keys(m).length}</span>`
      + (DEMO ? ` <span class="tag">Published build</span>` : "")) + (empty
    ? `<div class="state"><p class="state__title">No evaluation reports in the serving store</p><p class="meta">Run <code>make serving</code> after the milestones to load them.</p></div>`
    : `
    ${rt ? `<div class="kpis" style="margin-top:var(--s-4)">
      <div class="kpi"><span class="kpi__v" ${raw(rt["map@12"])}>${f5(rt["map@12"])}</span><span class="kpi__k">Ranker MAP@12 · test week</span></div>
      <div class="kpi"><span class="kpi__v">${liftA(rt, bt)}</span><span class="kpi__k">Ranker lift vs best baseline · test</span></div>
      ${rv ? `<div class="kpi"><span class="kpi__v" ${raw(rv["map@12"])}>${f5(rv["map@12"])}</span><span class="kpi__k">Ranker MAP@12 · validation week</span></div>` : ""}
      ${tb ? `<div class="kpi"><span class="kpi__v">${Object.keys(tb.results).length}</span><span class="kpi__k">Track B models compared · ${tbWeek} week</span></div>` : ""}
    </div>` : ""}
    <section class="studio-section" aria-labelledby="t-a"><h2 id="t-a">Track A: next-purchase ranking (MAP@12)</h2>
      ${table("Track A results", th(["system", "week", ["MAP@12"], ["returning"], ["new customers"], ["cold-item recall@12"]]), taRows)}
      <p class="notice">Ranker relative lift vs best baseline (repeat purchase + age-band popularity): validation ${liftA(rv, bv)}, test ${liftA(rt, bt)}</p>
    </section>
    <div class="studio-grid">
      <section class="studio-section" aria-labelledby="t-r"><h2 id="t-r">Retrieval recall (validation)</h2>
        ${table("Retrieval recall", th(["channel", ["recall"], ["cand./customer"], ["unique"]]), recRows)}</section>
      <section class="studio-section" aria-labelledby="t-f"><h2 id="t-f">Track B pair-mining funnel</h2>
        ${table("Track B funnel", th(["stage", ["count"]]), funnel)}</section>
    </div>
    <section class="studio-section" aria-labelledby="t-b"><h2 id="t-b">Track B: Complete the Look (${tbWeek} week)</h2>
      ${table("Track B results", th(["model", ["relative lift vs popularity"], ["Recall@12"], ["NDCG@12"], ["tail-item"], ["item cold start"], ["jewellery"], ["coverage"]]), tbRows)}
    </section>`));
  return root;
}

// ---- Serving --------------------------------------------------------------------------------------------
const when = t => t ? new Date(t * 1000).toLocaleString("en-GB", {dateStyle: "medium", timeStyle: "medium"}) : "—";
const labels = l => Object.entries(l || {}).map(([k, v]) => `${k}=${v}`).join(", ") || "—";

export async function serving(ctx) {
  ctx.setTitle("DS Studio · Serving");
  ctx.render(head("Serving status", "Loading the live serving bundle…") + loading);
  const settle = p => p.then(v => ({ok: true, v}), e => ({ok: false, e}));
  const [ready, meta, tel] = await Promise.all([
    settle(probe("/readyz").then(r => ({status: r.status, body: r.body}))),
    settle(api("/api/v2/meta")), settle(api("/api/v2/metrics"))]);
  if (!ctx.current()) return;
  const rd = ready.ok ? ready.v : null;
  const isReady = rd && rd.status === 200 && rd.body?.ready !== false;
  const mt = meta.ok ? meta.v : null, t = tel.ok ? tel.v : null;
  const kv = obj => Object.entries(obj).map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td class="num">${esc(typeof v === "object" ? JSON.stringify(v) : String(v))}</td></tr>`).join("");
  const unavailable = what => `<div class="notice notice--error" role="alert"><strong>${what} unavailable.</strong> The serving bundle may not be loaded on this server.</div>`;
  ctx.render(head("Serving status",
      DEMO
        ? "The bundle behind Complete the Look, visual search and outfit analysis in the local application. This page is a snapshot of that bundle's own status endpoints, captured when the published build was made; the counters below are the traffic that server process had seen at that moment, not an evaluation and not public traffic."
        : "The bundle behind live Complete the Look, visual search and outfit analysis. Telemetry below is live demo traffic on this server process since it started, not an evaluation.",
      isReady ? `<span class="tag tag--ok"><span class="status-dot is-ok" aria-hidden="true"></span>Ready</span>` : `<span class="tag tag--warn"><span class="status-dot is-bad" aria-hidden="true"></span>Not ready</span>`) + `
    <div class="kpis" style="margin-top:var(--s-4)">
      <div class="kpi"><span class="kpi__v">${isReady ? "Ready" : "Not ready"}</span><span class="kpi__k">/readyz${rd ? ` · HTTP ${rd.status}` : " · unreachable"}</span></div>
      ${mt ? `<div class="kpi"><span class="kpi__v">${esc(mt.serving_week)}</span><span class="kpi__k">Serving week · cutoff ${esc(mt.cutoff)}</span></div>
      <div class="kpi"><span class="kpi__v">${Number(mt.n_live_articles).toLocaleString()}</span><span class="kpi__k">Live articles</span></div>
      <div class="kpi"><span class="kpi__v">${esc(mt.equivalence?.identical_order_share)}</span><span class="kpi__k">Offline/online identical-order share (${esc(mt.equivalence?.requests)} requests)</span></div>
      <div class="kpi"><span class="kpi__v">${esc(mt.visual?.state ?? "—")}</span><span class="kpi__k">Visual model state</span></div>` : ""}
    </div>
    <div class="studio-grid">
      <section class="studio-section" aria-labelledby="sv-v"><h2 id="sv-v">Bundle and versions</h2>
        ${mt ? table("Bundle versions", th(["field", ["value"]]), kv({bundle_version: mt.bundle_version, model_version: mt.model_version,
          catalog_version: mt.catalog_version, profile_version: mt.profile_version, availability_version: mt.availability_version,
          pool: mt.pool, n_keys: mt.n_keys, personalization_enabled: mt.personalization_enabled, load_seconds: mt.load_seconds})) : unavailable("Bundle metadata")}
      </section>
      <section class="studio-section" aria-labelledby="sv-r"><h2 id="sv-r">Rankers</h2>
        ${mt ? table("Rankers", th(["ranker", ["trees"], ["features"]]), Object.entries(mt.rankers || {}).map(([k, v]) =>
          `<tr><th scope="row">${esc(k)}</th><td class="num">${esc(v.trees)}</td><td class="num">${esc(v.features)}</td></tr>`).join(""),
          "Equivalence check: " + esc(JSON.stringify(mt.equivalence?.tolerance ?? {}))) : unavailable("Ranker metadata")}
      </section>
    </div>
    <section class="studio-section" aria-labelledby="sv-t"><h2 id="sv-t">Demo telemetry <span class="tag">${DEMO ? "captured snapshot" : "live · this process"}</span></h2>
      ${t ? `<p class="caption">Cache: ${esc(t.cache?.hits)} hits, ${esc(t.cache?.misses)} misses, ${esc(t.cache?.size)} / ${esc(t.cache?.capacity)} entries.
          Last bundle load: ${esc(t.load_history?.at(-1)?.bundle_version ?? "—")} at ${when(t.load_history?.at(-1)?.at)}.</p>
        <div class="studio-grid">
          ${table("Counters", th(["counter", "labels", ["value"]]), t.counters.length ? t.counters.map(c =>
            `<tr><th scope="row">${esc(c.name)}</th><td>${esc(labels(c.labels))}</td><td class="num">${esc(c.value)}</td></tr>`).join("") : `<tr><td colspan="3">No traffic yet.</td></tr>`)}
          ${table("Latency", th(["histogram", "labels", ["count"], ["mean ms"]]), t.histograms.length ? t.histograms.map(h =>
            `<tr><th scope="row">${esc(h.name)}</th><td>${esc(labels(h.labels))}</td><td class="num">${esc(h.count)}</td><td class="num" ${raw(h.sum_ms)}>${h.count ? (h.sum_ms / h.count).toFixed(1) : "—"}</td></tr>`).join("") : `<tr><td colspan="4">No traffic yet.</td></tr>`)}
        </div>` : unavailable("Telemetry")}
    </section>`);
}

// ---- Inspect a recommendation -----------------------------------------------------------------------------
export async function inspect(ctx) {
  ctx.setTitle("DS Studio · Inspect");
  const customer = state.customer;
  ctx.render(head("Inspect a recommendation", "Loading this profile's recommendations…") + loading);
  const d = await getHome(customer);
  if (!ctx.current() || customer !== state.customer) return;
  const fy = d.modules.find(m => m.id === "for_you")?.items || [];
  const root = ctx.render(head("Inspect a recommendation",
      `Track A ranker output for the inspected profile, with the evidence-gated reasons and the SHAP contributions behind each score.
       Change the inspected profile with the profile control at the top right.`,
      `<span class="tag">Profile: ${esc(profileName(d.customer))} · ID ${d.customer.customer_idx}</span>`) + `
    <p class="meta">${esc(profileDetail(d.customer))} · segment <code>${esc(d.customer.segment)}</code></p>
    ${fy.length ? `<div class="inspect" style="margin-top:var(--s-5)">
      <ul class="inspect-list" role="list">${fy.map((it, i) => `<li><button class="inspect-item" type="button" aria-pressed="${i === 0}" data-i="${i}">
        <img src="${esc(it.image)}" alt="" width="1166" height="1750" loading="lazy">
        <span><span>${String(it.rank ?? i + 1).padStart(2, "0")} · ${esc(it.prod_name)}</span>
          <span class="meta">${esc(it.product_type_name)} · score <span class="tabular">${esc(it.score)}</span></span></span></button></li>`).join("")}</ul>
      <section class="studio-section" style="margin-top:0" id="inspect-detail" aria-live="polite"></section>
    </div>` : `<div class="state"><p class="state__title">No recommendations for this profile</p></div>`}`);
  if (!root || !fy.length) return;
  const show = async i => {
    const it = fy[i];
    $$(".inspect-item", root).forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.i === i)));
    const box = $("#inspect-detail", root);
    box.innerHTML = `<h2>${esc(it.prod_name)}</h2><p class="meta">Loading SHAP contributions…</p>`;
    try {
      const e = await api(`/api/explain/${customer}/${it.article_id}`);
      if (!ctx.current()) return;
      box.innerHTML = `<h2>${esc(it.prod_name)} <a class="link-btn" href="#/product/${it.article_id}">Open in Shop</a></h2>
        <p class="meta">Article ${it.article_id} · rank ${esc(it.rank)} · ranker score <span class="tabular">${esc(e.score)}</span></p>
        <h3 class="eyebrow">Evidence-gated reasons (DS only; not shown to shoppers)</h3>${reasonList(e.reasons)}
        <h3 class="eyebrow">SHAP contributions (top features)</h3>
        <p class="caption">Contribution of each feature to the LightGBM ranking score. Positive values push the item up.</p>${shapBars(e.shap)}`;
    } catch (err) {
      box.innerHTML = `<h2>${esc(it.prod_name)}</h2><div class="notice notice--error" role="alert">${esc(err.message)}</div>`;
    }
  };
  $(".inspect-list", root).addEventListener("click", e => { const b = e.target.closest(".inspect-item"); if (b) show(+b.dataset.i); });
  show(0);
}

// ---- Label ----------------------------------------------------------------------------------------------
export async function label(ctx) {
  const round = +(ctx.route.params.get("round") || 2);
  ctx.setTitle("Label");
  if (DEMO) {
    ctx.render(head("Label", "Not available in the published build.") + `
      <div class="state"><p class="state__title">This workflow runs locally only</p>
        <p class="meta">Labelling writes the human gold labels for the visual-search judge and shows crops of the
          evaluation outfit photos, which are not published. Run the application locally to use it.</p>
        <a class="btn btn--quiet" href="#/studio">Back to DS Studio</a></div>`);
    return;
  }
  ctx.render(head("Label", "Loading labelling tasks…") + loading);
  const tasks = await api(`/api/label/tasks?round=${round}`);
  if (!ctx.current()) return;
  const todo = tasks.filter(t => !t.done);
  const done = tasks.length - todo.length;
  if (!todo.length) {
    ctx.render(head(`Labelling done <span class="meta">(round ${round})</span>`, `All ${tasks.length} tasks are labelled. Thank you!`) +
      `<p><a class="btn btn--quiet" href="#/studio">Back to DS Studio</a></p>`);
    return;
  }
  const t = todo[0], sel = t.candidates.map(() => 0);
  const root = ctx.render(head(`Is it a reasonable substitute? <span class="meta">(round ${round})</span>`,
      `Task ${done + 1} of ${tasks.length}. For each product, select it if a shopper looking for the item on the left would accept it
       (same type of item, similar colour and style). Unselected = not relevant. These labels are the gold set for the visual-search judge.`) + `
    <div class="progress" role="progressbar" aria-label="Labelling progress" aria-valuemin="0" aria-valuemax="${tasks.length}" aria-valuenow="${done}"><span style="width:${100 * done / tasks.length}%"></span></div>
    <div class="label-view" style="margin-top:var(--s-5)">
      <figure class="label-crop" style="margin:0"><img src="${esc(t.crop)}" alt="Street photo crop of the ${esc(t.category)} to match">
        <figcaption class="meta">Street photo · ${esc(t.category)}</figcaption></figure>
      <div>
        <ul class="label-grid" role="list" aria-label="Candidate products">${t.candidates.map((c, i) => `<li>
          <button class="label-card" type="button" aria-pressed="false" data-i="${i}">
            <img src="${esc(c.image)}" alt="" width="1166" height="1750" loading="lazy">
            <span class="tile__name">${esc(c.prod_name)}</span>
            <span class="meta">${esc(c.product_type_name)} · ${esc(c.colour_group_name)}</span>
            <span class="label-card__state">Not relevant</span></button></li>`).join("")}</ul>
        <div class="label-actions">
          <button class="btn btn--solid" type="button" id="labsave">Save and next →</button>
          <span class="meta" id="lab-count" role="status">0 of ${t.candidates.length} selected as relevant</span>
          <span class="inline-status is-error" id="lab-err" role="alert"></span>
        </div>
      </div>
    </div>`);
  if (!root) return;
  $(".label-grid", root).addEventListener("click", e => {
    const b = e.target.closest(".label-card");
    if (!b) return;
    const i = +b.dataset.i;
    sel[i] = 1 - sel[i];
    b.setAttribute("aria-pressed", String(!!sel[i]));
    b.querySelector(".label-card__state").textContent = sel[i] ? "✓ Relevant" : "Not relevant";
    $("#lab-count", root).textContent = `${sel.reduce((a, x) => a + x, 0)} of ${t.candidates.length} selected as relevant`;
  });
  const save = $("#labsave", root);
  save.addEventListener("click", async () => {
    save.disabled = true;
    $("#lab-err", root).textContent = "";
    try {
      await api("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({task_id: t.task_id, relevant: sel, round})});
      if (ctx.current()) ctx.rerun();
    } catch (err) {
      $("#lab-err", root).textContent = `Not saved: ${err.message} Your selection is kept; try again.`;
      save.disabled = false;
    }
  });
}
