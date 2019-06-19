package Code;

import java.util.HashMap;
import java.util.LinkedHashSet;

public class LFUCache {
    private HashMap<Integer, Integer> values;
    private HashMap<Integer, Integer> counts;
    private HashMap<Integer, LinkedHashSet<Integer>> lists;
    private int capacity;
    private int min = -1;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        values = new HashMap<>();
        counts = new HashMap<>();
        lists = new HashMap<>();
        lists.put(1, new LinkedHashSet<>());
    }

    public int get(int key) {
        if (!values.containsKey(key)) return -1;
        int count = counts.get(key);
        counts.put(key, count + 1);
        lists.get(count).remove(key);
        if (count == min && lists.get(count).size() == 0) ++min;
        if (!lists.containsKey(count + 1)) {
            lists.put(count + 1, new LinkedHashSet<>());
        }
        lists.get(count + 1).add(key);
        return values.get(key);
    }

    public void put(int key, int value) {
        if (capacity <= 0) return;
        if (values.containsKey(key)) {
            values.put(key, value);
            get(key);
            return;
        }
        if (values.size() >= capacity) {
            int evit = lists.get(min).iterator().next();
            lists.get(min).remove(evit);
            values.remove(evit);
        }
        values.put(key, value);
        counts.put(key, 1);
        min = 1;
        lists.get(1).add(key);
    }
}
