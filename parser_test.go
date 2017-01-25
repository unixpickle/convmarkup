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
