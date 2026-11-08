import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import beta
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Моделювання читацької поведінки та інформаційних каскадів",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner=False)
def build_two_layer_reader_network(
    n_en: int = 300,
    n_ua: int = 500,
    m_edges: int = 4,
    p_triad: float = 0.35,
    alpha_bi: float = 0.14,
    alpha_loc: float = 0.68,
    omega_strong: float = 0.32,
    omega_weak: float = 0.08,
    eta_hub: float = 0.45,
    xi_res: float = 0.35,
    seed: int = 42
):
    """
    Генерація двосегментного кластеризованого графа G = (V, E, W)
    за моделлю Холма та Кіма (Holme-Kim) із білінгвальними мостами.
    """
    rng = np.random.default_rng(seed)
    g_en = nx.powerlaw_cluster_graph(n=n_en, m=m_edges, p=p_triad, seed=int(seed))
    g_ua = nx.powerlaw_cluster_graph(n=n_ua, m=m_edges, p=p_triad, seed=int(seed + 1))

    n_total = n_en + n_ua
    adj_strong = np.zeros((n_total, n_total), dtype=np.float64)
    adj_weak = np.zeros((n_total, n_total), dtype=np.float64)

    # Сегментація локальних вершин V_UA на V_bi, V_loc, V_inert
    n_bi = int(round(n_ua * alpha_bi))
    n_loc = int(round(n_ua * alpha_loc))
    ua_indices = np.arange(n_en, n_total)
    rng.shuffle(ua_indices)

    bi_nodes = ua_indices[:n_bi]
    loc_nodes = ua_indices[n_bi:n_bi + n_loc]
    inert_nodes = ua_indices[n_bi + n_loc:]

    node_segment = np.zeros(n_total, dtype=np.int32)
    # 0: EN, 1: UA_bi, 2: UA_loc, 3: UA_inert
    node_segment[bi_nodes] = 1
    node_segment[loc_nodes] = 2
    node_segment[inert_nodes] = 3

    # Виділення хабів (інфлюенсерів) за степенем вершин
    deg_en = np.array([g_en.degree(i) for i in range(n_en)], dtype=np.float64)
    deg_ua = np.array([g_ua.degree(i) for i in range(n_ua)], dtype=np.float64)
    k_min = max(1.0, float(min(deg_en.min(), deg_ua.min())))

    en_hubs = np.argsort(deg_en)[::-1][:max(5, int(0.05 * n_en))]
    ua_hubs = n_en + np.argsort(deg_ua)[::-1][:max(5, int(0.05 * n_ua))]

    # Формування внутрішніх зв'язків у глобальному контурі V_EN
    for u, v in g_en.edges():
        common = len(list(nx.common_neighbors(g_en, u, v)))
        union_sz = max(1, g_en.degree(u) + g_en.degree(v) - common - 2)
        overlap = common / union_sz
        for src, dst in [(u, v), (v, u)]:
            if src in en_hubs and overlap < 0.15:
                adj_weak[src, dst] = omega_weak * (1.0 + eta_hub * np.log(max(1.0, deg_en[src] / k_min)))
            else:
                adj_strong[src, dst] = omega_strong * (1.0 + overlap)

    # Додаткові односпрямовані ребра від англомовних хабів (BookTok / Goodreads)
    for h in en_hubs:
        targets = rng.choice(n_en, size=min(n_en // 3, int(deg_en[h] * 2)), replace=False)
        w_val = omega_weak * (1.0 + eta_hub * np.log(max(1.0, deg_en[h] / k_min)))
        for t_node in targets:
            if t_node != h and adj_strong[h, t_node] == 0:
                adj_weak[h, t_node] = w_val

    # Формування внутрішніх зв'язків у локальному контурі V_UA
    for u_local, v_local in g_ua.edges():
        common = len(list(nx.common_neighbors(g_ua, u_local, v_local)))
        union_sz = max(1, g_ua.degree(u_local) + g_ua.degree(v_local) - common - 2)
        overlap = common / union_sz
        u, v = n_en + u_local, n_en + v_local
        for src, dst, src_loc in [(u, v, u_local), (v, u, v_local)]:
            if src in ua_hubs and overlap < 0.15:
                adj_weak[src, dst] = omega_weak * (1.0 + eta_hub * np.log(max(1.0, deg_ua[src_loc] / k_min)))
            else:
                adj_strong[src, dst] = omega_strong * (1.0 + overlap)

    # Додаткові односпрямовані ребра від локальних книжкових блогерів
    for h in ua_hubs:
        h_loc = h - n_en
        targets = n_en + rng.choice(n_ua, size=min(n_ua // 3, int(deg_ua[h_loc] * 2)), replace=False)
        w_val = omega_weak * (1.0 + eta_hub * np.log(max(1.0, deg_ua[h_loc] / k_min)))
        for t_node in targets:
            if t_node != h and adj_strong[h, t_node] == 0:
                adj_weak[h, t_node] = w_val

    # Міжсегментні ребра E_cross (глобальний тренд -> локальна аудиторія)
    for b_node in bi_nodes:
        # Білінгви підписані на глобальних інфлюенсерів та англомовних читачів
        src_hubs = rng.choice(en_hubs, size=min(len(en_hubs), 3), replace=False)
        src_peers = rng.choice(n_en, size=2, replace=False)
        for s in src_hubs:
            adj_weak[s, b_node] = omega_weak * (1.0 + eta_hub * np.log(max(1.0, deg_en[s] / k_min)))
        for s in src_peers:
            adj_strong[s, b_node] = omega_strong * 0.85

    # Часткове проникнення алгоритмічних рекомендацій у масову аудиторію V_loc
    loc_exposed = rng.choice(loc_nodes, size=max(1, int(0.35 * len(loc_nodes))), replace=False)
    for l_node in loc_exposed:
        s = rng.choice(en_hubs)
        adj_weak[s, l_node] = omega_weak * 0.9

    # Стохастичне нормування вхідних ваг за формулою (2.17)
    in_sum = adj_strong.sum(axis=0) + adj_weak.sum(axis=0) + xi_res
    w_strong = adj_strong / in_sum[np.newaxis, :]
    w_weak = adj_weak / in_sum[np.newaxis, :]

    avg_clust_ua = nx.average_clustering(g_ua)
    cross_edges = int(np.count_nonzero(w_strong[:n_en, n_en:] + w_weak[:n_en, n_en:]))
    total_edges = int(np.count_nonzero(w_strong + w_weak))
    rho_cross = cross_edges / max(1, total_edges)

    return {
        "n_en": n_en,
        "n_ua": n_ua,
        "n_total": n_total,
        "w_strong": w_strong,
        "w_weak": w_weak,
        "node_segment": node_segment,
        "en_hubs": en_hubs,
        "ua_hubs": ua_hubs,
        "avg_clust_ua": avg_clust_ua,
        "rho_cross": rho_cross
    }


def run_single_simulation(
    net_data: dict,
    tau_lag: int,
    t_horizon: int,
    mu_theta_loc: float,
    delta_loc: float,
    lambda_weak: float,
    lambda_strong: float,
    nu_decay: float,
    n_ua_seed_hubs: int,
    rng: np.random.Generator
):
    """
    Один прогін дискретно-подійної симуляції модифікованої порогової моделі
    з динамічним бар'єром перекладу та релаксацією пам'яті.
    """
    n_en = net_data["n_en"]
    n_ua = net_data["n_ua"]
    n_total = net_data["n_total"]
    w_strong = net_data["w_strong"]
    w_weak = net_data["w_weak"]
    seg = net_data["node_segment"]

    # Генерація індивідуальних порогів за бета-розподілом (2.12)
    kappa = 12.0
    mu_bi = max(0.08, mu_theta_loc * 0.55)
    mu_inert = min(0.88, mu_theta_loc * 1.45)

    theta_base = np.zeros(n_total, dtype=np.float64)
    for s_code, mu_val in [(0, mu_theta_loc * 0.85), (1, mu_bi), (2, mu_theta_loc), (3, mu_inert)]:
        mask = (seg == s_code)
        a_param = max(1.1, mu_val * kappa)
        b_param = max(1.1, (1.0 - mu_val) * kappa)
        theta_base[mask] = beta.rvs(a_param, b_param, size=int(mask.sum()), random_state=rng)

    # Надбавка мовного бар'єра до релізу перекладу (2.23)
    delta_lang = np.zeros(n_total, dtype=np.float64)
    delta_lang[seg == 2] = delta_loc
    delta_lang[seg == 3] = 1.0

    # Вектор станів S_v(t) in {0, 1, 2} та час первинної активації t_u
    state = np.zeros(n_total, dtype=np.int32)
    t_act = np.full(n_total, -1, dtype=np.int32)
    max_pressure_pre_rel = np.zeros(n_total, dtype=np.float64)

    # Початковий імпульс у глобальному сегменті V_EN при t_0 = 0
    en_hubs = net_data["en_hubs"]
    init_en = np.concatenate([
        en_hubs[:max(3, len(en_hubs) // 2)],
        rng.choice(n_en, size=max(5, int(0.04 * n_en)), replace=False)
    ])
    init_en = np.unique(init_en)
    state[init_en] = 2
    t_act[init_en] = 0

    ua_slice = slice(n_en, n_total)
    v_ua_t = np.zeros(t_horizon + 1, dtype=np.float64)
    informed_ua_t = np.zeros(t_horizon + 1, dtype=np.float64)
    active_ua_t = np.zeros(t_horizon + 1, dtype=np.float64)
    active_en_t = np.zeros(t_horizon + 1, dtype=np.float64)

    active_en_t[0] = np.mean(state[:n_en] == 2)

    seeded_hubs_count = 0
    at_release_new_ua = 0

    for t in range(0, t_horizon):
        # Перевірка виходу перекладу та таргетованого посіву через локальні хаби
        if t + 1 == tau_lag and n_ua_seed_hubs > 0:
            ua_hubs = net_data["ua_hubs"]
            chosen_hubs = ua_hubs[:min(n_ua_seed_hubs, len(ua_hubs))]
            new_hubs = chosen_hubs[state[chosen_hubs] < 2]
            state[new_hubs] = 2
            t_act[new_hubs] = t + 1
            seeded_hubs_count = len(chosen_hubs)

        active_nodes = np.where(state == 2)[0]
        if len(active_nodes) > 0:
            dt = np.maximum(0.0, float(t) - t_act[active_nodes].astype(np.float64))
            decay_s = np.exp(-lambda_strong * (dt ** nu_decay))
            decay_w = np.exp(-lambda_weak * (dt ** nu_decay))

            # Обчислення соціального тиску I_v(t) за формулою (2.26)
            pressure = (
                decay_s @ w_strong[active_nodes, :] +
                decay_w @ w_weak[active_nodes, :]
            )
        else:
            pressure = np.zeros(n_total, dtype=np.float64)

        if t < tau_lag:
            max_pressure_pre_rel = np.maximum(max_pressure_pre_rel, pressure)

        # Динамічний поріг активації theta_v(t) за формулою (2.22)
        heaviside = 1.0 if (t + 1) >= tau_lag else 0.0
        theta_t = np.minimum(1.0, theta_base + delta_lang * (1.0 - heaviside))

        # Оновлення станів агентів (2.19) та (2.20)
        uninformed_mask = (state == 0) & (pressure > 0.005) & (pressure < theta_t)
        newly_active_mask = (state < 2) & (pressure >= theta_t)

        state[uninformed_mask] = 1
        state[newly_active_mask] = 2
        t_act[newly_active_mask] = t + 1

        new_ua = int(np.sum(newly_active_mask[ua_slice]))
        if t + 1 == tau_lag:
            at_release_new_ua = new_ua + seeded_hubs_count

        v_ua_t[t + 1] = new_ua / float(n_ua)
        informed_ua_t[t + 1] = np.mean(state[ua_slice] == 1)
        active_ua_t[t + 1] = np.mean(state[ua_slice] == 2)
        active_en_t[t + 1] = np.mean(state[:n_en] == 2)

    # Обчислення підсумкових метрик (2.31)-(2.37)
    ua_states = state[ua_slice]
    ua_t_act = t_act[ua_slice]
    ua_theta_base = theta_base[ua_slice]
    ua_max_pre = max_pressure_pre_rel[ua_slice]

    orig_mask = (ua_states == 2) & (ua_t_act >= 0) & (ua_t_act < tau_lag)
    trans_mask = (ua_states == 2) & (ua_t_act >= tau_lag)
    decay_lost_mask = (ua_states == 1) & (ua_max_pre >= ua_theta_base)

    sigma_orig = float(np.mean(orig_mask))
    sigma_trans = float(np.mean(trans_mask))
    sigma_decay = float(np.mean(decay_lost_mask))
    lambda_lost = sigma_orig + sigma_decay

    secondary_adopters = int(np.sum((ua_states == 2) & (ua_t_act > tau_lag)))
    init_trans_pool = max(1, at_release_new_ua)
    k_sec = float(secondary_adopters) / float(init_trans_pool)

    post_rel_rates = v_ua_t[tau_lag:] if tau_lag <= t_horizon else np.array([0.0])
    v_max_ua = float(np.max(post_rel_rates))
    t_peak_ua = int(tau_lag + np.argmax(post_rel_rates)) if tau_lag <= t_horizon else tau_lag

    return {
        "v_ua_t": v_ua_t,
        "informed_ua_t": informed_ua_t,
        "active_ua_t": active_ua_t,
        "active_en_t": active_en_t,
        "sigma_trans": sigma_trans,
        "sigma_orig": sigma_orig,
        "sigma_decay": sigma_decay,
        "lambda_lost": lambda_lost,
        "k_sec": k_sec,
        "v_max_ua": v_max_ua,
        "t_peak_ua": t_peak_ua
    }


@st.cache_data(show_spinner=False)
def run_monte_carlo_experiment(
    _net_data: dict,
    tau_lag: int,
    t_horizon: int,
    mu_theta_loc: float,
    delta_loc: float,
    lambda_weak: float,
    lambda_strong: float,
    nu_decay: float,
    n_ua_seed_hubs: int,
    n_runs: int = 50,
    base_seed: int = 100
):
    """
    Статистичне усереднення траєкторій та метрик за методом Монте-Карло (2.38).
    """
    results = []
    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r * 17)
        res = run_single_simulation(
            net_data=_net_data,
            tau_lag=tau_lag,
            t_horizon=t_horizon,
            mu_theta_loc=mu_theta_loc,
            delta_loc=delta_loc,
            lambda_weak=lambda_weak,
            lambda_strong=lambda_strong,
            nu_decay=nu_decay,
            n_ua_seed_hubs=n_ua_seed_hubs,
            rng=rng
        )
        results.append(res)

    keys_scalar = ["sigma_trans", "sigma_orig", "sigma_decay", "lambda_lost", "k_sec", "v_max_ua", "t_peak_ua"]
    summary = {}
    for k in keys_scalar:
        vals = np.array([r[k] for r in results], dtype=np.float64)
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if n_runs > 1 else 0.0

    for k_vec in ["v_ua_t", "informed_ua_t", "active_ua_t", "active_en_t"]:
        mat = np.array([r[k_vec] for r in results], dtype=np.float64)
        summary[f"{k_vec}_mean"] = np.mean(mat, axis=0)
        summary[f"{k_vec}_std"] = np.std(mat, axis=0, ddof=1) if n_runs > 1 else np.zeros(mat.shape[1])

    return summary


# Інтерфейс користувача Streamlit
st.title("📚 Імітаційне моделювання інформаційних каскадів на книжковому ринку")
st.markdown(
    "**Спеціальність 124 «Системний аналіз»** | Модифікована порогова модель (LTM) "
    "на двосегментному кластеризованому графі з динамічним бар'єром перекладу та згасанням уваги."
)

with st.sidebar:
    st.header("⚙️ Параметри системи")
    st.subheader("1. Топологія графа")
    n_ua_input = st.slider("Кількість вершин локального графа (N_UA)", 200, 1000, 500, 50)
    alpha_bi_input = st.slider("Частка ядра білінгвів (α_bi)", 0.04, 0.30, 0.14, 0.02)
    p_triad_input = st.slider("Імовірність тріади кластеризації (p_Δ)", 0.10, 0.70, 0.35, 0.05)

    st.subheader("2. Часові та видавничі параметри")
    tau_input = st.slider("Часовий лаг локалізації (τ, тижнів)", 0, 45, 12, 1)
    t_horizon_input = st.slider("Горизонт моделювання (T, тижнів)", 30, 65, 52, 2)
    n_seed_hubs_input = st.slider("Таргетовані локальні блогери (|S_hub|)", 0, 20, 2, 1)

    st.subheader("3. Поведінкові пороги та згасання")
    mu_theta_input = st.slider("Середній поріг жанрового скепсису (μ_θ)", 0.15, 0.65, 0.30, 0.05)
    delta_loc_input = st.slider("Мовний бар'єр масової аудиторії (δ_loc)", 0.20, 0.70, 0.42, 0.02)
    lambda_weak_input = st.slider("Швидкість згасання уваги у соцмережах (λ_weak)", 0.02, 0.25, 0.09, 0.01)
    n_mc_runs = st.slider("Кількість прогонів Монте-Карло (R)", 10, 100, 40, 10)

net = build_two_layer_reader_network(
    n_en=int(n_ua_input * 0.6),
    n_ua=n_ua_input,
    alpha_bi=alpha_bi_input,
    p_triad=p_triad_input
)

mc_res = run_monte_carlo_experiment(
    _net_data=net,
    tau_lag=tau_input,
    t_horizon=t_horizon_input,
    mu_theta_loc=mu_theta_input,
    delta_loc=delta_loc_input,
    lambda_weak=lambda_weak_input,
    lambda_strong=lambda_weak_input * 0.35,
    nu_decay=0.85,
    n_ua_seed_hubs=n_seed_hubs_input,
    n_runs=n_mc_runs
)

# Аналітична оцінка критичного вікна локалізації за формулою (2.29)
t_peak_global = 6.0
i_peak_est = 0.52
tau_crit_est = t_peak_global + (1.0 / lambda_weak_input) * np.log(max(1.01, i_peak_est / mu_theta_input))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Охоплення перекладу σ_trans(T)", f"{mc_res['sigma_trans_mean']*100:.1f}%", f"±{mc_res['sigma_trans_std']*100:.1f}%")
col2.metric("Відтік в оригінал σ_orig(T)", f"{mc_res['sigma_orig_mean']*100:.1f}%", f"±{mc_res['sigma_orig_std']*100:.1f}%")
col3.metric("Втрати згасання σ_decay(T)", f"{mc_res['sigma_decay_mean']*100:.1f}%", f"±{mc_res['sigma_decay_std']*100:.1f}%")
col4.metric("Вторинний каскад K_sec", f"{mc_res['k_sec_mean']:.2f}", f"±{mc_res['k_sec_std']:.2f}")
col5.metric("Критичне вікно τ_crit", f"{tau_crit_est:.1f} тиж.", f"Поточний лаг: {tau_input} тиж.")

tab1, tab2, tab3 = st.tabs([
    "📈 Динаміка інформаційного каскаду",
    "🔬 Порівняльний аналіз 4 сценаріїв (п. 3.3)",
    "🎯 Фазовий перехід від лагу τ та експорт даних"
])

with tab1:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    weeks = np.arange(t_horizon_input + 1)

    ax1.plot(weeks, mc_res["active_en_t_mean"] * 100, label="Глобальний контур V_EN (активні)", linestyle="--", color="#1f77b4", linewidth=2)
    ax1.plot(weeks, mc_res["active_ua_t_mean"] * 100, label="Локальний контур V_UA (покупці S=2)", color="#2ca02c", linewidth=2.5)
    ax1.fill_between(
        weeks,
        np.maximum(0, (mc_res["active_ua_t_mean"] - mc_res["active_ua_t_std"]) * 100),
        np.minimum(100, (mc_res["active_ua_t_mean"] + mc_res["active_ua_t_std"]) * 100),
        color="#2ca02c", alpha=0.18
    )
    ax1.plot(weeks, mc_res["informed_ua_t_mean"] * 100, label="Поінформовані в очікуванні (S=1)", color="#ff7f0e", linewidth=2)
    ax1.axvline(tau_input, color="#d62728", linestyle=":", linewidth=2, label=f"Реліз перекладу t_rel = {tau_input}")
    ax1.set_title("Кумулятивна динаміка станів читацької мережі")
    ax1.set_xlabel("Дискретний час t (тижні)")
    ax1.set_ylabel("Частка аудиторії (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5)

    ax2.bar(weeks, mc_res["v_ua_t_mean"] * 100, color="#6a5acd", alpha=0.8, label="Швидкість активації v_UA(t)")
    ax2.axvline(tau_input, color="#d62728", linestyle=":", linewidth=2, label="Вихід перекладу")
    ax2.set_title("Миттєва швидкість генерації попиту в Україні v_UA(t)")
    ax2.set_xlabel("Дискретний час t (тижні)")
    ax2.set_ylabel("Нові покупці за тиждень (% від N_UA)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)

    st.pyplot(fig)
    st.caption(
        f"Топологічні індикатори згенерованого графа: середній коефіцієнт кластеризації "
        f"C_avg = {net['avg_clust_ua']:.3f}, міжсегментна проникність ρ_cross = {net['rho_cross']:.4f}."
    )

with tab2:
    st.subheader("Експериментальне порівняння чотирьох базових сценаріїв (Розділ 3.3)")
    scenarios_cfg = [
        ("Сценарій 1: Хвиля резонансу (τ=4, α_bi=0.14, μ_θ=0.28, |S_hub|=2)", 4, 0.14, 0.28, 2),
        ("Сценарій 2: Виснаження та відтік ядра (τ=28, α_bi=0.24, μ_θ=0.28, |S_hub|=1)", 28, 0.24, 0.28, 1),
        ("Сценарій 3: Бар'єр жанрового скепсису (τ=12, α_bi=0.14, μ_θ=0.50, |S_hub|=2)", 12, 0.14, 0.50, 2),
        ("Сценарій 4: Компенсація через хаби (τ=28, α_bi=0.14, μ_θ=0.28, |S_hub|=12)", 28, 0.14, 0.28, 12),
    ]
    rows_scen = []
    fig_s, ax_s = plt.subplots(figsize=(10, 4.2))
    for name, s_tau, s_bi, s_mu, s_hub in scenarios_cfg:
        s_net = build_two_layer_reader_network(n_en=300, n_ua=500, alpha_bi=s_bi, p_triad=p_triad_input)
        s_res = run_monte_carlo_experiment(
            _net_data=s_net, tau_lag=s_tau, t_horizon=52, mu_theta_loc=s_mu,
            delta_loc=delta_loc_input, lambda_weak=lambda_weak_input,
            lambda_strong=lambda_weak_input * 0.35, nu_decay=0.85,
            n_ua_seed_hubs=s_hub, n_runs=25
        )
        ax_s.plot(np.arange(53), s_res["active_ua_t_mean"] * 100, linewidth=2.2, label=name.split("(")[0])
        rows_scen.append({
            "Сценарій": name,
            "Лаг τ (тиж.)": s_tau,
            "Частка перекладу σ_trans (%)": round(s_res["sigma_trans_mean"] * 100, 2),
            "Відтік в оригінал σ_orig (%)": round(s_res["sigma_orig_mean"] * 100, 2),
            "Втрати згасання σ_decay (%)": round(s_res["sigma_decay_mean"] * 100, 2),
            "Сумарні втрати Λ_lost (%)": round(s_res["lambda_lost_mean"] * 100, 2),
            "Коефіцієнт K_sec": round(s_res["k_sec_mean"], 2)
        })
    ax_s.set_title("Порівняння кумулятивних кривих залучення аудиторії за сценаріями 1–4")
    ax_s.set_xlabel("Час t (тижні)")
    ax_s.set_ylabel("Сукупна частка активованих читачів у V_UA (%)")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend()
    st.pyplot(fig_s)
    df_scen = pd.DataFrame(rows_scen)
    st.dataframe(df_scen, use_container_width=True)

with tab3:
    st.subheader("Залежність цільових метрик від тривалості видавничого лагу τ")
    tau_grid = [0, 4, 8, 12, 16, 20, 24, 30, 36, 42]
    sweep_rows = []
    for t_val in tau_grid:
        r_sw = run_monte_carlo_experiment(
            _net_data=net, tau_lag=t_val, t_horizon=52, mu_theta_loc=mu_theta_input,
            delta_loc=delta_loc_input, lambda_weak=lambda_weak_input,
            lambda_strong=lambda_weak_input * 0.35, nu_decay=0.85,
            n_ua_seed_hubs=n_seed_hubs_input, n_runs=20
        )
        sweep_rows.append({
            "tau_weeks": t_val,
            "sigma_trans_pct": round(r_sw["sigma_trans_mean"] * 100, 2),
            "sigma_orig_pct": round(r_sw["sigma_orig_mean"] * 100, 2),
            "sigma_decay_pct": round(r_sw["sigma_decay_mean"] * 100, 2),
            "lambda_lost_pct": round(r_sw["lambda_lost_mean"] * 100, 2),
            "k_sec": round(r_sw["k_sec_mean"], 2)
        })
    df_sweep = pd.DataFrame(sweep_rows)

    fig_p, ax_p = plt.subplots(figsize=(9.5, 4.0))
    ax_p.plot(df_sweep["tau_weeks"], df_sweep["sigma_trans_pct"], marker="o", linewidth=2.4, color="#2ca02c", label="Частка покупців перекладу σ_trans(T)")
    ax_p.plot(df_sweep["tau_weeks"], df_sweep["lambda_lost_pct"], marker="s", linewidth=2.2, color="#d62728", label="Сумарні втрати аудиторії Λ_lost(T)")
    ax_p.plot(df_sweep["tau_weeks"], df_sweep["sigma_orig_pct"], marker="^", linestyle="--", color="#1f77b4", label="Відтік у читання оригіналу σ_orig(T)")
    ax_p.axvline(tau_crit_est, color="#ff7f0e", linestyle=":", linewidth=2, label=f"Аналітичний поріг τ_crit ≈ {tau_crit_est:.1f} тиж.")
    ax_p.set_title("Фазовий перехід ефективності локалізації залежно від часового лагу τ")
    ax_p.set_xlabel("Тривалість затримки виходу перекладу τ (тижні)")
    ax_p.set_ylabel("Частка локальної мережі (%)")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend()
    st.pyplot(fig_p)

    csv_bytes = df_sweep.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Завантажити таблицю результатів експерименту (CSV)",
        data=csv_bytes,
        file_name="simulation_results_tau_sweep.csv",
        mime="text/csv"
    )
