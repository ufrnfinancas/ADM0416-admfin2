# Animação para a aula de Teoria de Carteiras: como a correlação muda a curva de combinações de dois ativos.
# Instalação: pip install manim   (requer também o ffmpeg e uma distribuição LaTeX, que você já tem)
# Renderização em alta qualidade: manim -qh manim_carteiras.py CurvaRho
# Pré-visualização rápida:        manim -pql manim_carteiras.py CurvaRho
# O vídeo final é salvo em C:\repo\ADM0416-admfin2\CurvaRho.mp4

from manim import *

# Fundo branco para combinar com o tema do beamer
config.background_color = WHITE

# Pasta de saída do vídeo e nome do arquivo
config.media_dir = r"C:\repo\ADM0416-admfin2"
config.video_dir = r"C:\repo\ADM0416-admfin2"
config.output_file = "CurvaRho"

# Valores ilustrativos, anualizados e nominais (substitua pelos obtidos no seu script de aula)
mu_A, sig_A = 0.25, 0.35
mu_B, sig_B = 0.12, 0.20


# O Manim exige uma classe Scene: é o único ponto do código que foge à sua regra de evitar classes
class CurvaRho(Scene):
    def construct(self):
        # Controle da correlação: tudo o que depende de rho é redesenhado a cada quadro
        rho = ValueTracker(1.0)
        # Peso do ativo A no ponto que percorre a curva
        peso = ValueTracker(0.5)

        # Fórmula do risco da carteira, no topo
        formula = MathTex(
            r"\sigma^2 = X_A^2\sigma_A^2 + X_B^2\sigma_B^2 + 2X_AX_B\sigma_A\sigma_B\rho_{A,B}",
            color=BLACK,
        ).scale(0.8).to_edge(UP)

        # Eixos: risco no horizontal e retorno esperado no vertical
        eixos = Axes(
            x_range=[0, 0.45, 0.05],
            y_range=[0, 0.30, 0.05],
            x_length=8.5,
            y_length=4.6,
            axis_config={"color": BLACK, "include_numbers": False, "include_tip": True},
        ).shift(DOWN * 0.7 + LEFT * 1.5)
        rotulo_x = Text("Risco", color=BLACK).scale(0.4).next_to(eixos.x_axis, DOWN, buff=0.2)
        rotulo_y = Text("Retorno esperado", color=BLACK).scale(0.4).next_to(eixos.y_axis, UP, buff=0.2)

        # Ativos A e B
        ponto_A = Dot(eixos.c2p(sig_A, mu_A), color=RED)
        ponto_B = Dot(eixos.c2p(sig_B, mu_B), color=RED)
        nome_A = Text("A", color=BLACK).scale(0.5).next_to(ponto_A, RIGHT)
        nome_B = Text("B", color=BLACK).scale(0.5).next_to(ponto_B, LEFT)

        # Curva de combinações: depende do valor atual de rho
        curva = always_redraw(lambda: ParametricFunction(
            lambda t: eixos.c2p(
                np.sqrt(max(t**2 * sig_A**2 + (1 - t) ** 2 * sig_B**2
                            + 2 * t * (1 - t) * sig_A * sig_B * rho.get_value(), 0.0)),
                t * mu_A + (1 - t) * mu_B,
                0,
            ),
            t_range=[0, 1, 0.01],
            color=BLUE,
            stroke_width=5,
        ))

        # Ponto móvel que mostra uma carteira específica sobre a curva
        ponto_carteira = always_redraw(lambda: Dot(
            eixos.c2p(
                np.sqrt(max(peso.get_value() ** 2 * sig_A**2 + (1 - peso.get_value()) ** 2 * sig_B**2
                            + 2 * peso.get_value() * (1 - peso.get_value()) * sig_A * sig_B * rho.get_value(), 0.0)),
                peso.get_value() * mu_A + (1 - peso.get_value()) * mu_B,
                0,
            ),
            color=GREEN_E,
        ))

        # Texto com o valor atual de rho, na lateral direita
        texto_rho = always_redraw(lambda: MathTex(
            r"\rho = " + f"{rho.get_value():.2f}".replace(".", ","), color=BLACK
        ).scale(1.2).to_edge(RIGHT).shift(UP * 0.5 + LEFT * 0.5))

        texto_peso = always_redraw(lambda: MathTex(
            r"X_A = " + f"{peso.get_value():.2f}".replace(".", ","), color=GREEN_E
        ).scale(0.9).next_to(texto_rho, DOWN, buff=0.4))

        # Sequência da animação
        self.play(Write(formula))
        self.play(Create(eixos), FadeIn(rotulo_x), FadeIn(rotulo_y))
        self.play(FadeIn(ponto_A), FadeIn(ponto_B), FadeIn(nome_A), FadeIn(nome_B))
        self.add(curva)
        self.play(FadeIn(texto_rho))
        self.wait(1)

        # Com rho = 1 a curva é uma reta entre A e B; o ponto percorre a reta
        self.add(ponto_carteira)
        self.play(FadeIn(texto_peso))
        self.play(peso.animate.set_value(0.0), run_time=2)
        self.play(peso.animate.set_value(1.0), run_time=3)
        self.play(peso.animate.set_value(0.5), run_time=2)
        self.wait(1)

        # Reduzindo a correlação, a curva se afasta da reta e o risco da carteira cai
        self.play(rho.animate.set_value(0.5), run_time=3)
        self.wait(1)
        self.play(rho.animate.set_value(0.0), run_time=3)
        self.wait(1)

        # Com rho = -1 a curva toca o eixo vertical: existe uma combinação de risco zero
        self.play(rho.animate.set_value(-1.0), run_time=3)
        self.wait(2)
