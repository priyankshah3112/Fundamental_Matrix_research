from FM.FM_main import fm
from GA.GA_main import ga


initial_population = fm()
final_population , stats = ga(initial_population)
print(final_population)

