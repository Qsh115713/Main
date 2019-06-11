package Data;

import java.util.*;

public class UndirectedGraphNode {
    public int label;
    public List<UndirectedGraphNode> neighbors;

    public UndirectedGraphNode(int label) {
        this.label = label;
        neighbors = new ArrayList<>();
    }
};
