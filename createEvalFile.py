import json
import re
def main():
    # Eval file path
  jsonFile = open("./comp-data.eval.json")
  uncompFile = open("./eval-data-uncomp.txt", mode ="+x")
  goldFile = open("./eval-data-gold.txt", mode ="+x")
  googleFile = open("./eval-data-google.txt", mode ="+x")
  uncompSentences = []
  goldStand = []
  googleStand = []


  jsonArray = jsonFile.read().split("\n\n")
  print(jsonArray[-1])
  # print(jsonArray[-1])
  for i, element in enumerate(jsonArray):
    try:
      data = json.loads(element)

      # Switched from .match() to .search() to search entire sentence str
      if not re.search("\d", data["graph"]["sentence"]):
        uncompSentences.append(data["graph"]["sentence"])
        googleStand.append(data["compression"]["text"])
        goldStand.append(data["headline"])

    except json.JSONDecodeError:
      print("error in google json")
    except KeyError as e:
          print(f"Error: Key not found. {e}")
    except Exception as e:
          print(f"An error occurred: {e}")
  print(len(uncompSentences))
  print(len(googleStand))
  print(len(goldStand))

  for sent in uncompSentences:
     uncompFile.write(sent)
     uncompFile.write("\n")
  for sent in goldStand:
     goldFile.write(sent)
     goldFile.write("\n")
  for sent in googleStand:
     googleFile.write(sent)
     googleFile.write("\n")


if __name__ == "__main__":
   main()