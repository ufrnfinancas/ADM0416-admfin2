# Animação detalhada da fronteira eficiente para a aula de Teoria de Carteiras.
# Renderização em alta qualidade: manim -qh manim_fronteira.py FronteiraEficiente
# Pré-visualização rápida:        manim -pql manim_fronteira.py FronteiraEficiente
# O vídeo final é salvo em C:\repo\ADM0416-admfin2\FronteiraEficiente.mp4

from manim import *
import numpy as np
from scipy.optimize import minimize

# Fundo branco para combinar com o tema do beamer
config.background_color = WHITE

# Pasta de saída do vídeo e nome do arquivo
config.media_dir = r"C:\repo\ADM0416-admfin2"
config.video_dir = r"C:\repo\ADM0416-admfin2"
config.output_file = "FronteiraEficiente"

# ----------------------------------------------------------------------------------------------
# Dados ilustrativos, anualizados e nominais (substitua pelos obtidos no seu script de aula)
# ----------------------------------------------------------------------------------------------
nomes = ["PETR4", "VALE3", "ITUB4", "WEGE3", "ABEV3"]
mu = np.array([0.28, 0.22, 0.18, 0.15, 0.12])
sig = np.array([0.35, 0.30, 0.25, 0.28, 0.20])

# Correlações geradas por um fator comum, o que garante matriz de covariância válida
carga = np.array([0.65, 0.60, 0.60, 0.50, 0.40])
corr = np.outer(carga, carga)
np.fill_diagonal(corr, 1.0)
cov = corr * np.outer(sig, sig)
n = len(mu)

limites = [(0.0, 1.0)] * n   # sem venda a descoberto
w0 = np.ones(n) / n
restr_soma = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

# Nuvem de carteiras aleatórias (conjunto de oportunidades)
rng = np.random.default_rng(7)
w_rand = rng.dirichlet(np.ones(n), size=400)
mu_rand = w_rand @ mu
sig_rand = np.sqrt(np.einsum("ij,jk,ik->i", w_rand, cov, w_rand))

# Conjunto de mínima variância: para cada retorno-alvo, a carteira de menor risco
alvos_g = np.linspace(mu.min(), mu.max(), 80)
sig_g = []
pesos_g = []
for alvo_i in alvos_g:
    restr = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
             {"type": "eq", "fun": lambda w, a=alvo_i: w @ mu - a}]
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=limites, constraints=restr)
    sig_g.append(np.sqrt(res.fun))
    pesos_g.append(res.x)
sig_g = np.array(sig_g)
pesos_g = np.array(pesos_g)

# Carteira de mínima variância global
res_mv = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=limites, constraints=restr_soma)
w_mv = res_mv.x
mu_mv = w_mv @ mu
sig_mv = np.sqrt(res_mv.fun)

# Parte eficiente (a partir da mínima variância global) e parte ineficiente
mascara_ef = alvos_g >= mu_mv
ret_ef_g = np.concatenate([[mu_mv], alvos_g[mascara_ef]])
sig_ef_g = np.concatenate([[sig_mv], sig_g[mascara_ef]])
pesos_ef_g = np.vstack([w_mv, pesos_g[mascara_ef]])
ret_inef_g = np.concatenate([alvos_g[~mascara_ef], [mu_mv]])
sig_inef_g = np.concatenate([sig_g[~mascara_ef], [sig_mv]])

# Carteira dominada, usada como exemplo: entre as carteiras aleatórias, a que está mais distante da fronteira,
# medida pelo menor entre a folga de retorno (mesmo risco) e a folga de risco (mesmo retorno), ambas normalizadas
ret_na_front_rand = np.interp(sig_rand, sig_ef_g, ret_ef_g)
sig_na_front_rand = np.interp(mu_rand, ret_ef_g, sig_ef_g)
folga_ret = (ret_na_front_rand - mu_rand) / (mu.max() - mu.min())
folga_risco = (sig_rand - sig_na_front_rand) / (sig.max() - sig.min())
valida = (mu_rand > mu_mv + 0.01) & (sig_rand > sig_mv) & (sig_rand < sig_ef_g.max())
escore = np.where(valida, np.minimum(folga_ret, folga_risco), -np.inf)
k_dom = int(np.argmax(escore))
mu_dom = mu_rand[k_dom]
sig_dom = sig_rand[k_dom]
mu_na_front = np.interp(sig_dom, sig_ef_g, ret_ef_g)      # mesmo risco, retorno da fronteira
sig_na_front = np.interp(mu_dom, ret_ef_g, sig_ef_g)      # mesmo retorno, risco da fronteira
print("Carteira dominada: retorno", mu_dom, "risco", sig_dom)
print("Na fronteira, mesmo risco: retorno", mu_na_front, "; mesmo retorno: risco", sig_na_front)

# Ativo livre de risco e carteira tangente
rf = 0.10
res_t = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), w0, method="SLSQP",
                 bounds=limites, constraints=restr_soma)
w_t = res_t.x
mu_t = w_t @ mu
sig_t = np.sqrt(w_t @ cov @ w_t)
sharpe_t = (mu_t - rf) / sig_t
x_max_reta = 0.39
sig_fim_capt = min(1.8 * sig_t, x_max_reta)

cores = [RED_D, BLUE_D, GREEN_D, ORANGE, PURPLE]


# O Manim exige uma classe Scene: é o único ponto do código que foge à sua regra de evitar classes
class FronteiraEficiente(Scene):
    def construct(self):
        # ------------------------------------------------------------------------------------------
        # Elementos fixos: título, eixos e rótulos
        # Layout: o gráfico ocupa a metade esquerda; a metade direita fica livre para legendas e para
        # o painel de composição da carteira. As legendas de rodapé têm no máximo cerca de 80 caracteres.
        # ------------------------------------------------------------------------------------------
        titulo = Text("Fronteira eficiente de Markowitz", color=BLACK).scale(0.7).to_edge(UP, buff=0.3)

        eixos = Axes(
            x_range=[0, 0.40, 0.05],
            y_range=[0, 0.32, 0.05],
            x_length=7.6,
            y_length=4.3,
            axis_config={"color": BLACK, "include_numbers": False, "include_tip": True},
        ).shift(LEFT * 2.4 + DOWN * 0.15)
        rotulo_x = Text("Risco (volatilidade)", color=BLACK).scale(0.4).next_to(eixos.x_axis, DOWN, buff=0.2)
        rotulo_x.align_to(eixos.x_axis, RIGHT)
        rotulo_y = Text("Retorno esperado", color=BLACK).scale(0.4).rotate(PI / 2)
        rotulo_y.next_to(eixos.y_axis, LEFT, buff=0.3)

        legenda = Text("Cada ponto é um ativo: risco e retorno esperado (valores ilustrativos).",
                       color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)

        self.play(Write(titulo))
        self.play(Create(eixos), FadeIn(rotulo_x), FadeIn(rotulo_y))
        self.play(FadeIn(legenda))

        # ------------------------------------------------------------------------------------------
        # Passo 1: os ativos individuais
        # ------------------------------------------------------------------------------------------
        pontos_ativos = VGroup(*[Dot(eixos.c2p(sig[i], mu[i]), color=cores[i], radius=0.09) for i in range(n)])
        nomes_ativos = VGroup(*[Text(nomes[i], color=BLACK).scale(0.35).next_to(pontos_ativos[i], RIGHT, buff=0.1)
                                for i in range(n)])
        self.play(LaggedStart(*[FadeIn(pontos_ativos[i]) for i in range(n)], lag_ratio=0.3))
        self.play(FadeIn(nomes_ativos))
        self.wait(1)

        # ------------------------------------------------------------------------------------------
        # Passo 2: a nuvem de carteiras possíveis
        # ------------------------------------------------------------------------------------------
        nova = Text("Variando os pesos, obtemos todas as carteiras possíveis.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        nuvem = VGroup(*[Dot(eixos.c2p(sig_rand[k], mu_rand[k]), color="#BBBBBB", radius=0.025)
                         for k in range(len(mu_rand))])
        self.play(ReplacementTransform(legenda, nova), FadeIn(nuvem, lag_ratio=0.01, run_time=3))
        legenda = nova
        self.bring_to_front(pontos_ativos, nomes_ativos)
        self.wait(1)

        # ------------------------------------------------------------------------------------------
        # Passo 3: o problema de otimização e o conjunto de mínima variância
        # ------------------------------------------------------------------------------------------
        problema = MathTex(
            r"\min_{w}\; w^{\top}\Sigma w \quad \text{sujeito a} \quad w^{\top}\mu = \mu_p,\;\; \sum_i w_i = 1,\;\; w_i \geq 0",
            color=BLACK,
        ).scale(0.7).next_to(titulo, DOWN, buff=0.2)
        nova = Text("Para cada retorno-alvo, buscamos a carteira de menor risco.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(Write(problema), ReplacementTransform(legenda, nova))
        legenda = nova

        curva_conj = VMobject(color=BLACK, stroke_width=4)
        curva_conj.set_points_smoothly([eixos.c2p(s, m, 0) for s, m in
                                        zip(np.concatenate([sig_inef_g, sig_ef_g[1:]]),
                                            np.concatenate([ret_inef_g, ret_ef_g[1:]]))])
        self.play(Create(curva_conj), run_time=4)
        nova = Text("O conjunto de mínima variância é a borda esquerda da nuvem.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(legenda, nova))
        legenda = nova
        self.wait(2)

        # ------------------------------------------------------------------------------------------
        # Passo 4: carteira de mínima variância global e separação entre parte ineficiente e eficiente
        # ------------------------------------------------------------------------------------------
        ponto_mv = Dot(eixos.c2p(sig_mv, mu_mv), color=RED, radius=0.11)
        nome_mv = Text("Mínima variância global", color=RED).scale(0.38).next_to(ponto_mv, LEFT, buff=0.15)
        nova = Text("Abaixo da mínima variância, há carteiras com mais risco e menos retorno.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ponto_mv, scale=2), FadeIn(nome_mv), ReplacementTransform(legenda, nova))
        legenda = nova

        curva_inef = VMobject(color=GRAY, stroke_width=4)
        curva_inef.set_points_smoothly([eixos.c2p(s, m, 0) for s, m in zip(sig_inef_g, ret_inef_g)])
        curva_inef_tracejada = DashedVMobject(curva_inef, num_dashes=30)
        curva_ef = VMobject(color=BLUE, stroke_width=7)
        curva_ef.set_points_smoothly([eixos.c2p(s, m, 0) for s, m in zip(sig_ef_g, ret_ef_g)])

        self.play(FadeOut(curva_conj), FadeIn(curva_inef_tracejada), Create(curva_ef), run_time=3)
        self.bring_to_front(ponto_mv, pontos_ativos, nomes_ativos)
        nome_ef = Text("Fronteira eficiente", color=BLUE).scale(0.45).next_to(eixos.c2p(0.25, 0.27), UP, buff=0.1)
        nome_inef = Text("Parte ineficiente", color=GRAY).scale(0.38).next_to(eixos.c2p(0.20, 0.10), RIGHT, buff=0.1)
        nova = Text("A fronteira eficiente é o trecho de cima: mais retorno para cada risco.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(nome_ef), FadeIn(nome_inef), ReplacementTransform(legenda, nova))
        legenda = nova
        self.play(FadeOut(problema))
        self.wait(2)

        # ------------------------------------------------------------------------------------------
        # Passo 5: dominância, usando a carteira igualmente ponderada
        # As explicações ficam na metade direita da tela, longe da nuvem de pontos.
        # ------------------------------------------------------------------------------------------
        ponto_eq = Dot(eixos.c2p(sig_dom, mu_dom), color=MAROON, radius=0.11)
        rotulos_dom = VGroup(
            Text("Carteira dominada (exemplo)", color=MAROON),
            Text("Mesmo risco, mais retorno", color=GREEN_E),
            Text("Mesmo retorno, menos risco", color=PURPLE),
        ).scale(0.4).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([4.8, 0.6, 0])

        nova = Text("Uma carteira fora da fronteira é dominada por alguma carteira da fronteira.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ponto_eq, scale=2), FadeIn(rotulos_dom[0]), ReplacementTransform(legenda, nova))
        legenda = nova

        seta_retorno = Arrow(eixos.c2p(sig_dom, mu_dom), eixos.c2p(sig_dom, mu_na_front), color=GREEN_E,
                             buff=0.08, stroke_width=5, max_tip_length_to_length_ratio=0.15)
        self.play(GrowArrow(seta_retorno), FadeIn(rotulos_dom[1]))
        self.wait(1)

        seta_risco = Arrow(eixos.c2p(sig_dom, mu_dom), eixos.c2p(sig_na_front, mu_dom), color=PURPLE,
                           buff=0.08, stroke_width=5, max_tip_length_to_length_ratio=0.15)
        self.play(GrowArrow(seta_risco), FadeIn(rotulos_dom[2]))
        self.wait(2)
        self.play(FadeOut(seta_retorno), FadeOut(seta_risco), FadeOut(ponto_eq), FadeOut(rotulos_dom))

        # ------------------------------------------------------------------------------------------
        # Passo 6: percorrendo a fronteira e vendo a composição da carteira
        # ------------------------------------------------------------------------------------------
        alvo = ValueTracker(mu_mv)
        base_y = -1.6
        x_barras = [3.0 + 0.85 * i for i in range(n)]

        titulo_painel = Text("Composição da carteira", color=BLACK).scale(0.45).move_to([4.8, 1.6, 0])
        eixo_barras = Line([2.6, base_y, 0], [7.0, base_y, 0], color=BLACK)
        nomes_barras = VGroup(*[Text(nomes[i], color=BLACK).scale(0.3).move_to([x_barras[i], base_y - 0.25, 0])
                                for i in range(n)])

        barras = always_redraw(lambda: VGroup(*[
            Rectangle(width=0.55,
                      height=max(np.interp(alvo.get_value(), ret_ef_g, pesos_ef_g[:, i]) * 3.0, 0.001),
                      fill_color=cores[i], fill_opacity=1, stroke_width=0)
            .move_to([x_barras[i], base_y, 0], aligned_edge=DOWN)
            for i in range(n)]))
        percentuais = always_redraw(lambda: VGroup(*[
            Text(f"{np.interp(alvo.get_value(), ret_ef_g, pesos_ef_g[:, i]) * 100:.0f}%", color=BLACK)
            .scale(0.32).move_to([x_barras[i], base_y - 0.6, 0])
            for i in range(n)]))

        ponto_movel = always_redraw(lambda: Dot(
            eixos.c2p(np.interp(alvo.get_value(), ret_ef_g, sig_ef_g), alvo.get_value()),
            color=GREEN_E, radius=0.12))
        leitura = always_redraw(lambda: Text(
            f"Retorno: {alvo.get_value() * 100:.1f}%   Risco: "
            f"{np.interp(alvo.get_value(), ret_ef_g, sig_ef_g) * 100:.1f}%".replace(".", ","),
            color=GREEN_E).scale(0.4).move_to([4.8, 1.1, 0]))

        nova = Text("Ao subir pela fronteira, a composição da carteira muda continuamente.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(nome_inef), FadeOut(nome_mv), ReplacementTransform(legenda, nova))
        legenda = nova
        self.add(barras, percentuais, ponto_movel, leitura)
        self.play(FadeIn(titulo_painel), FadeIn(eixo_barras), FadeIn(nomes_barras))
        self.wait(1)
        self.play(alvo.animate.set_value(mu.max()), run_time=8, rate_func=linear)
        self.wait(1)
        self.play(alvo.animate.set_value(mu_mv), run_time=4)
        nova = Text("Na base, o peso se espalha por vários ativos; no topo, vai todo para um só.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(legenda, nova))
        legenda = nova
        self.wait(3)

        # ------------------------------------------------------------------------------------------
        # Passo 7: ativo livre de risco, carteira tangente e nova fronteira
        # As explicações ficam na metade direita, em cores iguais às das linhas do gráfico.
        # ------------------------------------------------------------------------------------------
        self.play(FadeOut(barras), FadeOut(percentuais), FadeOut(ponto_movel), FadeOut(leitura),
                  FadeOut(titulo_painel), FadeOut(eixo_barras), FadeOut(nomes_barras), FadeOut(nome_ef))

        ponto_rf = Dot(eixos.c2p(0, rf), color=BLACK, radius=0.1)
        nome_rf = Text("Ativo livre de risco", color=BLACK).scale(0.38).next_to(ponto_rf, RIGHT, buff=0.15).shift(DOWN * 0.25)
        rotulos_rf = VGroup(
            Text("Carteira tangente", color=BLUE_E),
            Text("Aplicação à taxa livre de risco", color=GREEN_E),
            Text("Captação à taxa livre de risco", color=RED_D),
        ).scale(0.4).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([4.8, 0.6, 0])

        nova = Text("Com ativo livre de risco, combinamos a taxa livre de risco com ativos de risco.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ponto_rf, scale=2), FadeIn(nome_rf), ReplacementTransform(legenda, nova))
        legenda = nova

        ponto_t = Dot(eixos.c2p(sig_t, mu_t), color=BLUE_E, radius=0.12)
        reta_aplic = Line(eixos.c2p(0, rf), eixos.c2p(sig_t, mu_t), color=GREEN_E, stroke_width=6)
        reta_capt = Line(eixos.c2p(sig_t, mu_t),
                         eixos.c2p(sig_fim_capt, rf + sharpe_t * sig_fim_capt), color=RED_D, stroke_width=6)
        self.play(Create(reta_aplic), FadeIn(ponto_t, scale=2), FadeIn(rotulos_rf[0]), run_time=3)
        nova = Text("Entre o ativo livre de risco e a tangente, o investidor aplica.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(legenda, nova), FadeIn(rotulos_rf[1]))
        legenda = nova
        self.wait(2)
        self.play(Create(reta_capt), run_time=3)
        nova = Text("Além da tangente, o investidor capta à taxa livre de risco.",
                    color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(legenda, nova), FadeIn(rotulos_rf[2]))
        legenda = nova
        self.wait(2)

        texto_final = Text("A nova fronteira é a reta do ativo livre de risco que tangencia a fronteira.",
                           color=BLACK).scale(0.42).to_edge(DOWN, buff=0.3)
        self.play(ReplacementTransform(legenda, texto_final))
        self.wait(4)
