from neuron import h

soma = h.Section(name='soma')
soma.L = 10
soma.diam = 10
soma.nseg = 1

dend = h.Section(name='dend')
dend.L = 100
dend.diam = 2
dend.nseg = 1
dend.connect(soma(1))

axon = h.Section(name='axon')
axon.L = 100
axon.diam = 1
axon.nseg = 1
axon.connect(soma(0.))

h.topology()

h.finitialize()

for sec in h.allsec():
  for node in sec.allseg():
    print(node, "ri:", node.ri())

