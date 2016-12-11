package main

import (
	"math/rand"
	"time"

	"github.com/unixpickle/lightsout"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	var net *lightsout.Network
	var err error
	if net, err = lightsout.LoadNetwork("out_net"); err != nil {
		net = lightsout.NewNetwork()
	}
	net.Train(true)
	net.Save("out_net")
}
