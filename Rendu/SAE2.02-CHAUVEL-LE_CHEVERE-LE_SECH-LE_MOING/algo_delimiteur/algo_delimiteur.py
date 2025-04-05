from collections import deque

def testCorrespondanceDelimiteurs(S):
    L = []
    T = deque()
    c = 1
    n = len(S)

    delimiteursGauche = "([{"
    delimiteursDroit = ")]}"
    correspondance = {')': '(', ']': '[', '}': '{'}

    for i in range(n):
        char = S[i]
        
        if char in delimiteursGauche:
            L.append(c)
            T.append((char, c))
            c += 1
        
        elif char in delimiteursDroit:
            if not T: 
                return "" 

            gauche, d = T.pop()
            if correspondance[char] == gauche:
                L.append(d)
            else:
                return "" 

    if not T:
        return L
    
    return ""