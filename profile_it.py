#import cProfile
#cProfile.run('import main.py')

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from logpoly.model import Logpoly

logpoly = Logpoly()

with PyCallGraph(output=GraphvizOutput()):
    logpoly.fit(plot=True)