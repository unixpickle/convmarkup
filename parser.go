package convmarkup

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var (
	commandExpr = regexp.MustCompile(`^([A-Za-z]*)(\(([^\)]*)\))?( {)?$`)
	argExpr     = regexp.MustCompile(`^ *([A-Za-z]*)=([\-0-9\.]*) *$`)
)

// A ParseError is an error produced while trying to parse
// a piece of code.
type ParseError struct {
	Message string

	// Line is the line number, starting at 0.
	Line int
}

// Error produces an error message that incorporates the
// error message and line number.
func (p *ParseError) Error() string {
	return fmt.Sprintf("line %d: %s", p.Line+1, p.Message)
}

// ASTNode is a node in a parsed markup file.
//
// Each node corresponds to a single block.
//
// The root block has no attributes, nor does it have a
// block name.
type ASTNode struct {
	// Line is the line number, starting at 0.
	Line int

	BlockName string
	Attrs     map[string]float64
	Children  []*ASTNode
}

// Parse converts a string of code into a root ASTNode for
// a markup file.
func Parse(contents string) (*ASTNode, error) {
	lines := strings.Split(contents, "\n")
	for i, x := range lines {
		y := strings.TrimSpace(x)
		if strings.HasPrefix(y, "#") {
			lines[i] = ""
		} else {
			lines[i] = y
		}
	}
	parsed, err := parseLines(0, lines)
	if err != nil {
		return nil, err
	}
	return &ASTNode{Children: parsed}, nil
}

// parseLines parses a list of lines.
func parseLines(off int, l []string) ([]*ASTNode, error) {
	var res []*ASTNode
	for i := 0; i < len(l); i++ {
		x := l[i]
		if x == "" {
			continue
		}
		parsed := commandExpr.FindStringSubmatch(x)
		if parsed == nil {
			return nil, &ParseError{
				Message: "invalid block declaration",
				Line:    off + i,
			}
		}
		name := parsed[1]
		attrs, err := parseAttrs(parsed[3])
		if err != nil {
			return nil, &ParseError{
				Message: err.Error(),
				Line:    off + i,
			}
		}
		node := &ASTNode{
			Line:      off + i,
			BlockName: name,
			Attrs:     attrs,
		}
		if parsed[4] != "" {
			closeIdx, err := matchingClose(l, i)
			if err != nil {
				return nil, &ParseError{
					Message: err.Error(),
					Line:    off + i,
				}
			}
			node.Children, err = parseLines(off+i+1, l[i+1:closeIdx])
			if err != nil {
				return nil, err
			}
			i = closeIdx
		}
		res = append(res, node)
	}
	return res, nil
}

// matchingClose finds the matching close curly-brace for
// the line at the given index.
func matchingClose(lines []string, open int) (int, error) {
	numIndent := 1
	for i := open + 1; i < len(lines); i++ {
		l := lines[i]
		if l == "}" {
			numIndent--
			if numIndent == 0 {
				return i, nil
			}
		} else if strings.HasSuffix(l, "{") {
			numIndent++
		}
	}
	return 0, errors.New("no matching }")
}

// parseAttrs parses an attribute list.
func parseAttrs(str string) (map[string]float64, error) {
	res := map[string]float64{}
	if str == "" {
		return res, nil
	}
	for i, x := range strings.Split(str, ",") {
		parsed := argExpr.FindStringSubmatch(x)
		if parsed == nil {
			return nil, fmt.Errorf("bad format for attribute %d", i)
		}
		name := parsed[1]
		value, err := strconv.ParseFloat(parsed[2], 64)
		if err != nil {
			return nil, fmt.Errorf("bad format for attribute %d", i)
		}
		if _, ok := res[name]; ok {
			return nil, fmt.Errorf("duplicate attribute: %s", name)
		}
		res[name] = value
	}
	return res, nil
}
