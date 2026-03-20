from neo4j import GraphDatabase

# 🔧 UPDATE THESE VALUES
uri = "neo4j+ssc://bbfc15b4.databases.neo4j.io"
user = "bbfc15b4"
password = "XbWsALChx59peicCmqfarprZfj5VLVr3blNsERJn25U"


def test_connection():
    print("\n🔍 Testing Neo4j Connection...")
    print("URI:", uri)
    print("USER:", user)
    print("PASSWORD LENGTH:", len(password))

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            result = session.run("RETURN 'Connection Successful' AS message")
            record = result.single()

            print("\n✅ SUCCESS:", record["message"])

        driver.close()

    except Exception as e:
        print("\n❌ ERROR:", str(e))


if __name__ == "__main__":
    test_connection()