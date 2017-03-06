package convmarkup

import (
	"reflect"
	"testing"
)

func TestParse(t *testing.T) {
	code := `# This is a neural net.
		Input(w=224, h=224,d=3.0)
		ReLU
		# Commented line
		MyBlock(attr=1) {
			First
			#Commented line
			Child {
				NamedBlock()
			}
			Another(a=3)
		}
	`
	actual, err := Parse(code)
	if err != nil {
		t.Fatal(err)
	}
	expected := &ASTNode{
		Children: []*ASTNode{
			{
				Line:      1,
				BlockName: "Input",
				Attrs:     map[string]float64{"w": 224, "h": 224, "d": 3},
			},
			{
				Line:      2,
				BlockName: "ReLU",
				Attrs:     map[string]float64{},
			},
			{
				Line:      4,
				BlockName: "MyBlock",
				Attrs:     map[string]float64{"attr": 1},
				Children: []*ASTNode{
					{
						Line:      5,
						BlockName: "First",
						Attrs:     map[string]float64{},
					},
					{
						Line:      7,
						BlockName: "Child",
						Attrs:     map[string]float64{},
						Children: []*ASTNode{
							{
								Line:      8,
								BlockName: "NamedBlock",
								Attrs:     map[string]float64{},
							},
						},
					},
					{
						Line:      10,
						BlockName: "Another",
						Attrs:     map[string]float64{"a": 3},
					},
				},
			},
		},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %#v but got %#v", expected, actual)
	}
}

func TestParseErrors(t *testing.T) {
	invalid := []string{
		"MyBlock(a=a)",
		"MyBlock(b=3....14)",
		"MyBlock=2",
		"MyBlock{\n}",
		"MyBlock #comment",
	}
	for i, x := range invalid {
		if _, err := Parse(x); err == nil {
			t.Errorf("sample %d should have failed", i)
		}
	}
}

func TestASTNodeBlock(t *testing.T) {
	markup := `
	Input(w=224, h=113, d=3)

	Padding(l=2, r=0, t=1, b=3)
	Conv(w=3, h=5, n=64, sx=2, sy=4)
	BatchNorm
	ReLU

	MaxPool(w=1, h=2)
	Residual {
	 	Padding(l=1, r=1, t=1, b=1)
	 	Conv(w=3, h=3, n=64)
	}
	Residual {
	 	Projection {
	 		Conv(w=1, h=1, n=128)
	 	}
		Repeat(n=2) {
			Padding(l=1, r=1, t=1, b=1)
	 		Conv(w=3, h=3, n=64)
		}
		Resize(w=114, h=16)
		Conv(w=3, h=3, n=128)
	}

	Assert(w=112, h=14, d=128)
	MeanPool(w=2, h=3, sx=1, sy=2)
	FC(out=10)
	Softmax
	Sigmoid
	Tanh
	Linear(scale=10, bias=5)
	`

	parsed, err := Parse(markup)
	if err != nil {
		t.Fatal(err)
	}

	actual, err := parsed.Block(Dims{}, DefaultCreators())
	if err != nil {
		t.Fatal(err)
	}
	expected := &Root{
		Children: []Block{
			&Input{Out: Dims{Width: 224, Height: 113, Depth: 3}},
			&Padding{Top: 1, Right: 0, Bottom: 3, Left: 2,
				Out: Dims{Width: 226, Height: 117, Depth: 3}},
			&Conv{FilterWidth: 3, FilterHeight: 5, FilterCount: 64, StrideX: 2, StrideY: 4,
				Out: Dims{Width: 112, Height: 29, Depth: 64}},
			&Activation{Name: "BatchNorm", Out: Dims{Width: 112, Height: 29, Depth: 64}},
			&Activation{Name: "ReLU", Out: Dims{Width: 112, Height: 29, Depth: 64}},
			&Pool{Name: "MaxPool", Width: 1, Height: 2, StrideX: 1, StrideY: 2,
				Out: Dims{Width: 112, Height: 14, Depth: 64}},
			&Residual{Residual: []Block{
				&Padding{Left: 1, Right: 1, Top: 1, Bottom: 1,
					Out: Dims{Width: 114, Height: 16, Depth: 64}},
				&Conv{FilterWidth: 3, FilterHeight: 3, FilterCount: 64, StrideX: 1,
					StrideY: 1, Out: Dims{Width: 112, Height: 14, Depth: 64}},
			}},
			&Residual{Projection: []Block{
				&Conv{FilterWidth: 1, FilterHeight: 1, FilterCount: 128, StrideX: 1,
					StrideY: 1, Out: Dims{Width: 112, Height: 14, Depth: 128}},
			}, Residual: []Block{
				&Repeat{N: 2, In: Dims{Width: 112, Height: 14, Depth: 64},
					Children: []Block{&Padding{Left: 1, Right: 1, Top: 1, Bottom: 1,
						Out: Dims{Width: 114, Height: 16, Depth: 64}},
						&Conv{FilterWidth: 3, FilterHeight: 3, FilterCount: 64, StrideX: 1,
							StrideY: 1, Out: Dims{Width: 112, Height: 14, Depth: 64}}}},
				&Resize{Out: Dims{Width: 114, Height: 16, Depth: 64}},
				&Conv{FilterWidth: 3, FilterHeight: 3, FilterCount: 128, StrideX: 1,
					StrideY: 1, Out: Dims{Width: 112, Height: 14, Depth: 128}},
			}},
			&Assert{In: Dims{Width: 112, Height: 14, Depth: 128}},
			&Pool{Name: "MeanPool", Width: 2, Height: 3, StrideX: 1, StrideY: 2,
				Out: Dims{Width: 111, Height: 6, Depth: 128}},
			&FC{OutCount: 10},
			&Activation{Name: "Softmax", Out: Dims{Width: 1, Height: 1, Depth: 10}},
			&Activation{Name: "Sigmoid", Out: Dims{Width: 1, Height: 1, Depth: 10}},
			&Activation{Name: "Tanh", Out: Dims{Width: 1, Height: 1, Depth: 10}},
			&Linear{Scale: 10, Bias: 5, In: Dims{Width: 1, Height: 1, Depth: 10}},
		},
	}

	aRoot, ok := actual.(*Root)
	if !ok {
		t.Fatalf("root should be Root but it's %T", aRoot)
	}

	if len(aRoot.Children) != len(expected.Children) {
		t.Fatalf("expected %d children but got %d", len(expected.Children),
			len(aRoot.Children))
	}

	for i, x := range expected.Children {
		a := aRoot.Children[i]
		if !reflect.DeepEqual(a, x) {
			t.Errorf("child %d: expected %#v but got %#v", i, x, a)
		}
	}
}

func TestASTnodeFailures(t *testing.T) {
	input := "Input(w=224, h=224, d=3)\n"
	invalid := []string{
		input + "Residual {\n}",
		input + "Padding(l=1, r=1, t=3)",
		input + "MaxPool(w=2)",
		input + "MeanPool(w=2)",
		input + "MeanPool(w=2, h=2, sx=0)",
		input + "Conv(w=3, h=2)",
		input + "Residual {\nConv(w=3, h=3, n=3)\n}",
		input + "Residual {\nConv(w=1, h=1, n=5)\n}",
		input + "Residual {\nProjection {\nConv(w=1, h=1, n=5)\n}\nConv(w=1, h=1, n=3)\n}",
		input + "Residual {\nProjection {\n\n}\nConv(w=1, h=1, n=3)\n}",
		input + "FC",
		input + "Assert(w=223, h=224, d=3)",
		input + "Assert(w=224, h=223, d=3)",
		input + "Assert(w=224, h=224, d=2)",
		input + "Repeat(n=0)",
		input + "Repeat(n=1) {\nConv(w=3, h=3, n=3)\n}",
		input + "Repeat(n=1) {\nConv(w=1, h=1, n=4)\n}",
		input + "Linear(foo=1)",
	}
	for i, x := range invalid {
		parsed, err := Parse(x)
		if err != nil {
			t.Errorf("parse %d: %s", i, err)
			continue
		}
		_, err = parsed.Block(Dims{}, DefaultCreators())
		if err == nil {
			t.Errorf("test %d did not fail", i)
		}
	}
}
