package Data;

import java.util.List;

public class TempNode {
    public int val;
    public List<TempNode> neighbors;

    public TempNode() {}

    public TempNode(int _val, List<TempNode> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
