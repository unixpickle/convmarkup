package convmarkup

import (
	"errors"
	"fmt"
)

// ErrUnsupportedBlock is used by functions in a Realizer.
var ErrUnsupportedBlock = errors.New("unsupported block type")

// A Realizer instantiates Blocks, typically only of a
// certain variety.
//
// A Realizer is passed a RealizerChain which can be used
// to instantiate sub-blocks.
//
// When a Realizer does not support a Block, it should
// return ErrUnsupportedBlock.
//
// A Realizer may return (nil, nil) to indicate that the
// Block has no meaningful instantiation.
// This is appropriate for blocks such as Input or Assert.
type Realizer interface {
	Realize(chain RealizerChain, inDims Dims, b Block) (interface{}, error)
}

// MetaRealizer is a Relaizer for the meta-blocks Assert
// and Input.
type MetaRealizer struct{}

// For meta-blocks, (nil, nil) is returned.
// Otherwise (nil, ErrUnsupportedBlock) is returned.
func (m MetaRealizer) Realize(chain RealizerChain, inDims Dims,
	b Block) (interface{}, error) {
	switch b.(type) {
	case *Assert, *Input:
		return nil, nil
	default:
		return nil, ErrUnsupportedBlock
	}
}

// A RealizerChain realizes Blocks by trying one Realizer
// at a time, in order, until one works.
type RealizerChain []Realizer

// Realize attempts to realize the Block by trying each
// Realizer in order.
// If no Realizer supports the Block, a detailed error is
// returned.
// The supported return value is false if and only if all
// the Realizers returned ErrUnsupportedBlock.
func (r RealizerChain) Realize(d Dims, b Block) (val interface{}, supported bool,
	err error) {
	for _, realizer := range r {
		obj, err := realizer.Realize(r, d, b)
		if err == ErrUnsupportedBlock {
			continue
		}
		return obj, true, err
	}
	return nil, false, fmt.Errorf("unsupported block: %T", b)
}
