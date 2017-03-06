package convmarkup

import (
	"errors"
	"fmt"
)

// Common errors during parsing.
var (
	ErrUnexpectedChildren = errors.New("unexpected children")
	ErrNotEnoughChildren  = errors.New("not enough children")
)

// Dims defines the dimensions of a 3D tensor.
type Dims struct {
	Width  int
	Height int
	Depth  int
}

// A Block is a concrete instance of a block.
type Block interface {
	Type() string
	OutDims() Dims
}

// A Creator can create blocks.
type Creator func(in Dims, attr map[string]float64, children []Block) (Block, error)

// DefaultCreators returns a mapping from block names to
// creators.
func DefaultCreators() map[string]Creator {
	return map[string]Creator{
		"":           CreateRoot,
		"Input":      CreateInput,
		"Assert":     CreateAssert,
		"Conv":       CreateConv,
		"Padding":    CreatePadding,
		"Resize":     CreateResize,
		"Residual":   CreateResidual,
		"Projection": CreateProjection,
		"FC":         CreateFC,
		"Repeat":     CreateRepeat,
		"Linear":     CreateLinear,
		"MaxPool":    PoolCreator("MaxPool"),
		"MeanPool":   PoolCreator("MeanPool"),
		"BatchNorm":  ActivationCreator("BatchNorm"),
		"ReLU":       ActivationCreator("ReLU"),
		"Sigmoid":    ActivationCreator("Sigmoid"),
		"Tanh":       ActivationCreator("Tanh"),
		"Softmax":    ActivationCreator("Softmax"),
	}
}

// Root is a root block.
//
// A Root must always have at least one child.
type Root struct {
	Children []Block
}

// CreateRoot creates a Root block.
func CreateRoot(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) == 0 {
		return nil, ErrNotEnoughChildren
	}
	if err := hasAllAndOnlyInts(attr, 0); err != nil {
		return nil, err
	}
	return &Root{Children: children}, nil
}

// Type returns the empty string.
func (r *Root) Type() string {
	return ""
}

// OutDims returns the dims from the last child.
func (r *Root) OutDims() Dims {
	return r.Children[len(r.Children)-1].OutDims()
}

// Input is the block that describes the input dimensions.
type Input struct {
	Out Dims
}

// CreateInput creates an *Input block.
func CreateInput(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) != 0 {
		return nil, ErrUnexpectedChildren
	}
	if err := hasAllAndOnlyInts(attr, 1, "w", "h", "d"); err != nil {
		return nil, err
	}
	return &Input{Out: Dims{
		Width:  int(attr["w"]),
		Height: int(attr["h"]),
		Depth:  int(attr["d"]),
	}}, nil
}

// Type returns "Input".
func (i *Input) Type() string {
	return "Input"
}

// OutDims returns i.Out.
func (i *Input) OutDims() Dims {
	return i.Out
}

// Assert is a place-holder block that ensures a specific
// tensor dimension.
type Assert struct {
	In Dims
}

// CreateAssert creates an *Assert block.
func CreateAssert(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) != 0 {
		return nil, ErrUnexpectedChildren
	}
	if err := hasAllAndOnlyInts(attr, 0, "w", "h", "d"); err != nil {
		return nil, err
	}
	if int(attr["w"]) != in.Width || int(attr["h"]) != in.Height ||
		int(attr["d"]) != in.Depth {
		return nil, fmt.Errorf("expected dimensions %dx%dx%d but got %dx%dx%d",
			int(attr["w"]), int(attr["h"]), int(attr["d"]),
			in.Width, in.Height, in.Depth)
	}
	return &Assert{In: in}, nil
}

// Type returns "Assert".
func (a *Assert) Type() string {
	return "Assert"
}

// OutDims returns the output dimensions.
func (a *Assert) OutDims() Dims {
	return a.In
}

// Conv is a Block for a convolutional layer.
type Conv struct {
	FilterWidth  int
	FilterHeight int
	FilterCount  int

	StrideX int
	StrideY int

	Out Dims
}

// CreateConv creates a (Conv block.
func CreateConv(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) > 0 {
		return nil, ErrUnexpectedChildren
	}
	if err := onlyTheseAttrs(attr, "w", "h", "n", "sx", "sy"); err != nil {
		return nil, err
	}
	if err := hasAllAttrs(attr, "w", "h", "n"); err != nil {
		return nil, err
	}
	if err := validInt(attr, 1, "w", "h", "n", "sx", "sy"); err != nil {
		return nil, err
	}

	res := &Conv{
		FilterWidth:  int(attr["w"]),
		FilterHeight: int(attr["h"]),
		FilterCount:  int(attr["n"]),
		StrideX:      int(attr["sx"]),
		StrideY:      int(attr["sy"]),
	}

	if res.StrideX == 0 {
		res.StrideX = 1
	}
	if res.StrideY == 0 {
		res.StrideY = 1
	}

	res.Out = Dims{
		Width:  1 + (in.Width-res.FilterWidth)/res.StrideX,
		Height: 1 + (in.Height-res.FilterHeight)/res.StrideY,
		Depth:  res.FilterCount,
	}
	if res.Out.Width < 0 {
		res.Out.Width = 0
	}
	if res.Out.Height < 0 {
		res.Out.Height = 0
	}

	return res, nil
}

// Type returns "Conv".
func (c *Conv) Type() string {
	return "Conv"
}

// OutDims returns the output dimensions.
func (c *Conv) OutDims() Dims {
	return c.Out
}

// Pool is a pooling block.
// The Name attribute will be "MaxPool" or "MeanPool".
type Pool struct {
	Name    string
	Width   int
	Height  int
	StrideX int
	StrideY int
	Out     Dims
}

// PoolCreator makes a Creator for a pool type.
func PoolCreator(name string) Creator {
	return func(in Dims, attr map[string]float64, children []Block) (Block, error) {
		if len(children) > 0 {
			return nil, ErrUnexpectedChildren
		}
		if err := onlyTheseAttrs(attr, "w", "h", "sx", "sy"); err != nil {
			return nil, err
		}
		if err := hasAllAttrs(attr, "w", "h"); err != nil {
			return nil, err
		}
		if err := validInt(attr, 1, "w", "h", "sx", "sy"); err != nil {
			return nil, err
		}
		res := &Pool{
			Name:    name,
			Width:   int(attr["w"]),
			Height:  int(attr["h"]),
			StrideX: int(attr["sx"]),
			StrideY: int(attr["sy"]),
		}
		if res.StrideX == 0 {
			res.StrideX = res.Width
		}
		if res.StrideY == 0 {
			res.StrideY = res.Height
		}
		res.Out = Dims{
			Width:  1 + (in.Width-res.Width)/res.StrideX,
			Height: 1 + (in.Height-res.Height)/res.StrideY,
			Depth:  in.Depth,
		}
		if res.Out.Width < 0 {
			res.Out.Width = 0
		}
		if res.Out.Height < 0 {
			res.Out.Height = 0
		}
		return res, nil
	}
}

// Type returns p.Name.
func (p *Pool) Type() string {
	return p.Name
}

// OutDims returns the output dimensions.
func (p *Pool) OutDims() Dims {
	return p.Out
}

// Padding is a tensor padding block.
type Padding struct {
	Top    int
	Right  int
	Bottom int
	Left   int
	Out    Dims
}

// CreatePadding creates a *Padding block.
func CreatePadding(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) > 0 {
		return nil, ErrUnexpectedChildren
	}
	if err := hasAllAndOnlyInts(attr, 0, "t", "r", "b", "l"); err != nil {
		return nil, err
	}
	res := &Padding{
		Top:    int(attr["t"]),
		Right:  int(attr["r"]),
		Bottom: int(attr["b"]),
		Left:   int(attr["l"]),
	}
	res.Out = Dims{
		Width:  in.Width + res.Left + res.Right,
		Height: in.Height + res.Top + res.Bottom,
		Depth:  in.Depth,
	}
	return res, nil
}

// Type returns "Padding".
func (p *Padding) Type() string {
	return "Padding"
}

// OutDims returns the output dimensions.
func (p *Padding) OutDims() Dims {
	return p.Out
}

// Resize is a tensor resizing block.
type Resize struct {
	Out Dims
}

// CreateResize creates a *Resize block.
func CreateResize(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) > 0 {
		return nil, ErrUnexpectedChildren
	}
	if err := hasAllAndOnlyInts(attr, 1, "w", "h"); err != nil {
		return nil, err
	}
	if in.Width == 0 || in.Height == 0 || in.Depth == 0 {
		return nil, errors.New("input cannot be empty")
	}
	res := &Resize{
		Out: Dims{
			Width:  int(attr["w"]),
			Height: int(attr["h"]),
			Depth:  in.Depth,
		},
	}
	return res, nil
}

// Type returns "Resize".
func (r *Resize) Type() string {
	return "Resize"
}

// OutDims returns the output dimensions.
func (r *Resize) OutDims() Dims {
	return r.Out
}

// Residual is a residual layer.
type Residual struct {
	// Projection may be nil.
	Projection []Block

	Residual []Block
}

// CreateResidual creates a *Residual block.
func CreateResidual(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) < 1 {
		return nil, ErrNotEnoughChildren
	}
	var projChildren []Block
	projBlock, ok := children[0].(*Projection)
	if ok {
		projChildren = projBlock.Children
		children = children[1:]
	}
	if len(children) < 1 {
		return nil, ErrNotEnoughChildren
	}
	layersOut := children[len(children)-1].OutDims()
	if ok {
		projOut := projChildren[len(projChildren)-1].OutDims()
		if projOut != layersOut {
			return nil, errors.New("residual output size mismatch")
		}
	} else {
		if layersOut != in {
			return nil, errors.New("residual output size mismatch")
		}
	}
	return &Residual{
		Projection: projChildren,
		Residual:   children,
	}, nil
}

// Type returns "Residual".
func (r *Residual) Type() string {
	return "Residual"
}

// OutDims returns the output dimensions.
func (r *Residual) OutDims() Dims {
	return r.Residual[len(r.Residual)-1].OutDims()
}

// Projection is a meta-block for Residual blocks.
type Projection struct {
	Children []Block
	In       Dims
}

// CreateProjection creates a *Projection block.
func CreateProjection(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if err := hasAllAndOnlyInts(attr, 0); err != nil {
		return nil, err
	}
	if len(children) == 0 {
		return nil, ErrNotEnoughChildren
	}
	return &Projection{
		Children: children,
		In:       in,
	}, nil
}

// Type returns "Projection".
func (p *Projection) Type() string {
	return "Projection"
}

// OutDims returns the projection's input dimensions so
// that the projection block can be followed by the
// contents of a Residual.
func (p *Projection) OutDims() Dims {
	return p.In
}

// FC is a fully-connected layer.
type FC struct {
	OutCount int
}

// CreateFC creates an *FC block.
func CreateFC(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if len(children) != 0 {
		return nil, ErrUnexpectedChildren
	}
	if err := hasAllAndOnlyInts(attr, 1, "out"); err != nil {
		return nil, err
	}
	return &FC{OutCount: int(attr["out"])}, nil
}

// Type returns "FC".
func (f *FC) Type() string {
	return "FC"
}

// OutDims returns the output dimensions.
func (f *FC) OutDims() Dims {
	return Dims{Width: 1, Height: 1, Depth: f.OutCount}
}

// Repeat is a meta-block for repeating its contents.
type Repeat struct {
	N        int
	Children []Block
	In       Dims
}

// CreateRepeat creates a *Repeat block.
func CreateRepeat(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if err := hasAllAndOnlyInts(attr, 1, "n"); err != nil {
		return nil, err
	}
	if len(children) > 0 {
		if children[len(children)-1].OutDims() != in {
			return nil, errors.New("input and output lengths must match")
		}
	}
	return &Repeat{
		N:        int(attr["n"]),
		Children: children,
		In:       in,
	}, nil
}

// Type returns "Repeat".
func (r *Repeat) Type() string {
	return "Repeat"
}

// OutDims returns the Repeat's output dimensions.
func (r *Repeat) OutDims() Dims {
	return r.In
}

// Linear is a block for scaling and biasing the input
// tensor.
type Linear struct {
	Scale float64
	Bias  float64
	In    Dims
}

// CreateLinear creates a *Scale block.
func CreateLinear(in Dims, attr map[string]float64, children []Block) (Block, error) {
	if err := onlyTheseAttrs(attr, "scale", "bias"); err != nil {
		return nil, err
	}
	if len(children) > 0 {
		return nil, ErrUnexpectedChildren
	}
	res := &Linear{
		Scale: attr["scale"],
		Bias:  attr["bias"],
		In:    in,
	}
	if _, ok := attr["scale"]; !ok {
		res.Scale = 1
	}
	return res, nil
}

// Type returns "Linear".
func (l *Linear) Type() string {
	return "Linear"
}

// OutDims returns the Linear's output dimensions.
func (l *Linear) OutDims() Dims {
	return l.In
}

// Activation is any block with no attributes.
type Activation struct {
	Name string
	Out  Dims
}

// ActivationCreator makes a Creator that creates
// activation blocks of the given type.
func ActivationCreator(name string) Creator {
	return func(in Dims, a map[string]float64, c []Block) (Block, error) {
		if len(c) != 0 {
			return nil, ErrUnexpectedChildren
		}
		if err := hasAllAndOnlyInts(a, 0); err != nil {
			return nil, err
		}
		return &Activation{Name: name, Out: in}, nil
	}
}

// Type returns a.Name.
func (a *Activation) Type() string {
	return a.Name
}

// OutDims returns a.Out.
func (a *Activation) OutDims() Dims {
	return a.Out
}

func hasAllAndOnlyInts(attrs map[string]float64, min int, allowed ...string) error {
	if err := onlyTheseAttrs(attrs, allowed...); err != nil {
		return err
	}
	if err := hasAllAttrs(attrs, allowed...); err != nil {
		return err
	}
	return validInt(attrs, min, allowed...)
}

func onlyTheseAttrs(attrs map[string]float64, allowed ...string) error {
	for a := range attrs {
		has := false
		for _, x := range allowed {
			if x == a {
				has = true
				break
			}
		}
		if !has {
			return errors.New("unexpected attribute: " + a)
		}
	}
	return nil
}

func hasAllAttrs(attrs map[string]float64, mustHave ...string) error {
	for _, x := range mustHave {
		if _, ok := attrs[x]; !ok {
			return errors.New("missing attribute: " + x)
		}
	}
	return nil
}

func validInt(attrs map[string]float64, min int, names ...string) error {
	for _, name := range names {
		val, ok := attrs[name]
		if ok {
			if val != float64(int(val)) {
				return errors.New("attribute " + name + " must be integer")
			} else if int(val) < min {
				return fmt.Errorf("attribute %s cannot be %d", name, int(val))
			}
		}
	}
	return nil
}
