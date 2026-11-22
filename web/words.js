/**
 * SignSpeak - Smart Predictive Word & Phrase Dictionary
 * =====================================================
 * Frequency-ranked dictionary with multi-word phrase completions.
 */

// Priority high-frequency communicative words
const HIGH_PRIORITY_WORDS = [
  "i", "you", "help", "need", "want", "please", "thank", "hello", "good",
  "yes", "no", "can", "water", "food", "pain", "doctor", "family", "call",
  "time", "where", "what", "how", "name", "feel", "love", "friend", "home",
  "hungry", "tired", "sick", "stop", "go", "come", "see", "hear", "understand"
];

// Conversational common phrases for auto-suggest
const COMMON_PHRASES = [
  "thank you", "please help", "how are you", "i need help", "good morning",
  "good night", "excuse me", "i am hungry", "i am thirsty", "call 911",
  "call my family", "i feel sick", "where is the bathroom", "nice to meet you"
];

const COMMON_WORDS = [
  "a", "about", "above", "across", "action", "activity", "actually", "add", "address",
  "after", "again", "against", "age", "ago", "agree", "air", "all", "allow", "almost",
  "alone", "along", "already", "also", "always", "am", "among", "and", "another",
  "answer", "any", "anyone", "anything", "appear", "apple", "are", "area", "arm",
  "around", "arrive", "art", "as", "ask", "at", "attack", "attention", "aunt", "away",
  "baby", "back", "bad", "bag", "ball", "bank", "base", "be", "bear", "beat",
  "beautiful", "because", "become", "bed", "before", "begin", "behind", "believe",
  "bell", "best", "better", "between", "big", "bird", "birth", "bit", "black", "blood",
  "blow", "blue", "board", "boat", "body", "book", "born", "both", "bottle", "bottom",
  "box", "boy", "bread", "break", "breakfast", "bridge", "bright", "bring", "brother",
  "brown", "build", "burn", "bus", "business", "busy", "but", "buy", "by", "cake",
  "call", "calm", "came", "camera", "camp", "can", "cannot", "car", "card", "care",
  "careful", "carry", "case", "cat", "catch", "cause", "center", "certain", "chair",
  "chance", "change", "charge", "check", "child", "children", "choose", "church", "city",
  "clean", "clear", "climb", "clock", "close", "cloth", "clothes", "cloud", "coffee",
  "cold", "color", "come", "comfort", "common", "company", "complete", "computer",
  "condition", "continue", "control", "cook", "cool", "corner", "correct", "cost",
  "could", "count", "country", "course", "cover", "cow", "cross", "cry", "cup", "cut",
  "dad", "dance", "danger", "dark", "day", "dead", "deal", "dear", "death", "decide",
  "deep", "dentist", "describe", "desk", "detail", "did", "die", "different", "difficult",
  "dinner", "direct", "direction", "dirty", "discover", "dish", "do", "doctor", "dog",
  "door", "double", "doubt", "down", "draw", "dream", "dress", "drink", "drive", "drop",
  "dry", "during", "each", "ear", "early", "earth", "east", "easy", "eat", "education",
  "effect", "egg", "eight", "either", "electric", "elephant", "else", "empty", "end",
  "enemy", "enjoy", "enough", "enter", "equal", "escape", "even", "evening", "event",
  "ever", "every", "everyone", "exact", "example", "except", "excuse", "expect", "expensive",
  "experience", "explain", "eye", "face", "fact", "fall", "family", "famous", "far",
  "farm", "fast", "fat", "father", "fear", "feel", "feeling", "few", "field", "fight",
  "figure", "fill", "film", "final", "find", "fine", "finger", "finish", "fire", "first",
  "fish", "fit", "five", "fix", "flag", "flat", "floor", "fly", "follow", "food",
  "foot", "for", "force", "foreign", "forest", "forget", "form", "forward", "four",
  "free", "fresh", "friend", "from", "front", "full", "fun", "game", "garden", "gas",
  "general", "gentle", "get", "girl", "give", "glad", "glass", "go", "god", "gold",
  "good", "goodbye", "got", "govern", "grand", "grass", "great", "green", "ground",
  "group", "grow", "guess", "guide", "gun", "hair", "half", "hand", "happen", "happy",
  "hard", "has", "hat", "have", "he", "head", "hear", "heart", "heat", "heavy", "help",
  "her", "here", "high", "hill", "him", "his", "history", "hit", "hold", "hole",
  "holiday", "home", "hope", "horse", "hospital", "hot", "hotel", "hour", "house",
  "how", "huge", "human", "hundred", "hungry", "hurt", "husband", "ice", "idea", "if",
  "ill", "important", "in", "include", "increase", "indeed", "inside", "instead", "interest",
  "into", "iron", "is", "island", "issue", "it", "its", "job", "join", "journey", "joy",
  "judge", "jump", "just", "keep", "key", "kill", "kind", "king", "kitchen", "knee",
  "knife", "know", "knowledge", "lady", "lake", "land", "large", "last", "late", "laugh",
  "law", "lay", "lead", "leader", "learn", "leave", "left", "leg", "less", "lesson",
  "let", "letter", "level", "lie", "life", "light", "like", "line", "lip", "listen",
  "little", "live", "lock", "long", "look", "lose", "lot", "loud", "love", "low",
  "machine", "main", "make", "man", "many", "map", "mark", "market", "marry", "matter",
  "may", "maybe", "me", "meal", "mean", "measure", "meat", "medicine", "meet", "member",
  "memory", "men", "middle", "might", "milk", "mind", "mine", "minute", "miss", "mistake",
  "mix", "moment", "money", "month", "moon", "more", "morning", "most", "mother",
  "mountain", "mouth", "move", "much", "music", "must", "my", "myself", "name", "nation",
  "nature", "near", "nearly", "neck", "need", "neighbor", "neither", "nerve", "never",
  "new", "news", "next", "nice", "night", "nine", "no", "nobody", "noise", "none",
  "noon", "nor", "north", "nose", "not", "note", "nothing", "notice", "now", "number",
  "nurse", "object", "ocean", "of", "off", "offer", "office", "often", "oil", "old",
  "on", "once", "one", "only", "open", "operation", "opinion", "or", "orange", "order",
  "other", "our", "out", "outside", "over", "own", "page", "pain", "paint", "pair",
  "paper", "parent", "park", "part", "party", "pass", "past", "path", "patient", "pay",
  "peace", "pen", "pencil", "people", "per", "perfect", "perhaps", "period", "person",
  "phone", "picture", "piece", "place", "plan", "plant", "play", "pleasant", "please",
  "plenty", "point", "police", "poor", "popular", "position", "possible", "post", "pot",
  "power", "prepare", "present", "press", "pretty", "price", "print", "prison", "private",
  "probable", "problem", "produce", "promise", "proper", "protect", "proud", "prove",
  "provide", "public", "pull", "pure", "push", "put", "quarter", "queen", "question",
  "quick", "quiet", "quite", "radio", "rain", "raise", "reach", "read", "ready", "real",
  "reason", "receive", "record", "red", "regard", "regular", "relation", "remember",
  "remove", "repeat", "reply", "report", "require", "rest", "result", "return", "rich",
  "ride", "right", "ring", "rise", "river", "road", "rock", "roll", "roof", "room",
  "root", "rope", "rough", "round", "row", "rule", "run", "safe", "sail", "same",
  "sand", "save", "say", "school", "science", "sea", "search", "season", "seat", "second",
  "secret", "see", "seed", "seem", "self", "sell", "send", "sense", "sentence", "separate",
  "serve", "service", "set", "settle", "seven", "several", "shall", "share", "sharp",
  "she", "sheet", "ship", "shirt", "shoe", "shoot", "shop", "short", "should", "shoulder",
  "shout", "show", "shut", "sick", "side", "sight", "sign", "silence", "silent", "silver",
  "simple", "since", "sing", "single", "sister", "sit", "six", "size", "skin", "sky",
  "sleep", "slip", "slow", "small", "smell", "smile", "smoke", "snow", "so", "soap",
  "soft", "soil", "soldier", "some", "someone", "something", "sometimes", "son", "song",
  "soon", "sorry", "sound", "south", "space", "speak", "special", "speed", "spell",
  "spend", "spirit", "sport", "spread", "spring", "square", "stage", "stand", "star",
  "start", "state", "station", "stay", "steam", "steel", "step", "stick", "still", "stone",
  "stop", "store", "story", "straight", "strange", "street", "stretch", "strong", "student",
  "study", "subject", "substance", "such", "sudden", "sugar", "suggest", "suit", "summer",
  "sun", "supper", "supply", "support", "sure", "surface", "surprise", "sweet", "swim",
  "system", "table", "tail", "take", "talk", "tall", "taste", "tax", "tea", "teach",
  "teacher", "team", "tear", "tell", "ten", "term", "test", "than", "thank", "that",
  "the", "their", "them", "then", "there", "these", "they", "thick", "thin", "thing",
  "think", "third", "this", "those", "though", "thought", "three", "through", "throw",
  "thus", "tie", "tight", "till", "time", "tire", "to", "today", "together", "told",
  "tomorrow", "too", "took", "tooth", "top", "touch", "toward", "town", "track", "train",
  "travel", "tree", "trip", "trouble", "true", "trust", "truth", "try", "turn", "twelve",
  "twenty", "two", "type", "under", "understand", "unit", "until", "up", "upon", "us",
  "use", "usual", "valley", "value", "various", "vegetable", "very", "village", "voice",
  "visit", "wait", "walk", "wall", "want", "war", "warm", "wash", "waste", "watch",
  "water", "wave", "way", "we", "weak", "wear", "weather", "week", "weigh", "welcome",
  "well", "went", "were", "west", "wet", "what", "wheel", "when", "where", "which",
  "while", "white", "who", "whole", "whose", "why", "wide", "wife", "wild", "will",
  "win", "wind", "window", "wine", "winter", "wire", "wise", "wish", "with", "without",
  "woman", "wonder", "wood", "word", "work", "world", "worry", "worse", "worth", "would",
  "wound", "write", "wrong", "yard", "year", "yellow", "yes", "yesterday", "yet", "you",
  "young", "your"
];

/**
 * Returns prioritized suggestions matching the prefix.
 */
function getWordSuggestions(prefix, maxResults = 6) {
  if (!prefix || prefix.trim().length === 0) {
    return ["I", "Help", "Need", "Want", "Please", "Thank you"].slice(0, maxResults);
  }

  const clean = prefix.trim().toLowerCase();
  const results = [];

  // 1. Check matching conversational phrases first
  for (const phrase of COMMON_PHRASES) {
    if (phrase.startsWith(clean) && phrase !== clean) {
      results.push(phrase);
      if (results.length >= maxResults) return results;
    }
  }

  // 2. High priority words
  for (const w of HIGH_PRIORITY_WORDS) {
    if (w.startsWith(clean) && w !== clean && !results.includes(w)) {
      results.push(w);
      if (results.length >= maxResults) return results;
    }
  }

  // 3. General dictionary
  for (const w of COMMON_WORDS) {
    if (w.startsWith(clean) && w !== clean && !results.includes(w)) {
      results.push(w);
      if (results.length >= maxResults) return results;
    }
  }

  // 4. Substring matches if still space
  if (results.length < maxResults) {
    for (const w of COMMON_WORDS) {
      if (w.includes(clean) && w !== clean && !results.includes(w)) {
        results.push(w);
        if (results.length >= maxResults) return results;
      }
    }
  }

  return results;
}
