# Formula da iteração: Epsilon final = Epsilon inicial * x^n
# Recompensa do Episodio 974

epsilon = 0.2324868256432007
count = 0
while epsilon > 0.05:
    epsilon = epsilon * 0.998503255
    count += 1

print(count + 974)