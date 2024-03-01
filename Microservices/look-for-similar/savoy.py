from unidecode import unidecode


class Savoy:

    def __finalVowelPortuguese(self, word):
        len_word = len(word)
        if len_word > 3:
            if word[-1] == 'e' or word[-1] == 'a' or word[-1] == 'o':
                word = word[:-1]

        return word

    def __remove_PTsuffix(self, word):
        len_word = len(word)

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'e' and (
                    word[-3] == 'r' or word[-3] == 's' or word[-3] == 'z' or word[-3] == 'l'):
                word = word[:-2]
                return word
        if len_word > 2:
            if word[-1] == 's' and word[-2] == 'n':
                new_word = list(word)
                new_word[-2] = 'm'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if (word[-1] == 's' and word[-2] == 'i') and (word[-3] == 'e' or word[-3] == 'é'):
                new_word = list(word)
                new_word[-3] = 'e'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'i' and word[-3] == 'a':
                new_word = list(word)
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'i' and word[-3] == 'ó':
                new_word = list(word)
                new_word[-3] = 'o'
                new_word[-2] = 'l'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 3:
            if word[-1] == 's' and word[-2] == 'i':
                new_word = list(word)
                new_word[-1] = 'l'
                sing = "".join(new_word)
                return sing

        if len_word > 2:
            if word[-1] == 's' and word[-2] == 'e' and word[-3] == 'õ':
                new_word = list(word)
                new_word[-3] = 'ã'
                new_word[-2] = 'o'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing
            if word[-1] == 's' and word[-2] == 'e' and word[-3] == 'ã':
                new_word = list(word)
                new_word[-2] = 'o'
                sing = "".join(new_word)
                sing = sing[:-1]
                return sing

        if len_word > 5:
            if word[-1] == 'e' and word[-2] == 't' and word[-3] == 'n' and word[-4] == 'e' and word[-5] == 'm':
                word = word[:-5]
                return word

        if len_word > 2:
            if word[-1] == 's':
                word = word[:-1]

        return word

    def __normFemininPortuguese(self, word):

        len_word = len(word)

        if len_word < 3 or word[-1] != 'a':
            return word

        if len_word > 6:

            if word[-2] == 'h' and word[-3] == 'n' and word[-4] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'c' and word[-3] == 'a' and word[-4] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'r' and word[-3] == 'i' and word[-4] == 'e':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

        if len_word > 5:
            if word[-2] == 'n' and word[-3] == 'o':
                new_word = list(word)
                new_word[-3] = 'ã'
                new_word[-2] = 'o'
                masc = "".join(new_word)
                masc = masc[:-1]
                return masc

            if word[-2] == 'r' and word[-3] == 'o':
                word = word[:-1]
                return word

            if word[-2] == 's' and word[-3] == 'o':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 's' and word[-3] == 'e':
                new_word = list(word)
                new_word[-3] = 'ê'
                masc = "".join(new_word)
                masc = masc[:-1]
                return masc

            if word[-2] == 'c' and word[-3] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'd' and word[-3] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'd' and word[-3] == 'a':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'v' and word[-3] == 'i':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'm' and word[-3] == 'a':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

            if word[-2] == 'n':
                new_word = list(word)
                new_word[-1] = 'o'
                masc = "".join(new_word)
                return masc

        return word

    def stem(self, word):
        len_word = len(word)
        if len_word > 2:
            word = self.__remove_PTsuffix(word)
            word = self.__normFemininPortuguese(word)
            word = self.__finalVowelPortuguese(word)
            word = unidecode(word)

        return word
